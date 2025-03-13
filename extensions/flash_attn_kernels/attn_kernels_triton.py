import torch
import triton
import triton.language as tl
import time

@triton.jit
def causal_flash_attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    start_kv,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    is_causal: tl.constexpr,
    attn_mask_ptr=None,
):
    K_block = tl.load(K_block_ptr)
    QK_block = tl.dot(Q_block, K_block)

    if is_causal:
        # Apply causal mask
        causal_mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
        
        # Apply softmax scaling and causal masking
        QK_block = QK_block * softmax_scale
        
        # If attention mask is provided, apply it in addition to causal mask
        if attn_mask_ptr is not None:
            # Load the attention mask for the current block
            attn_mask_block = tl.load(attn_mask_ptr)
            # Combine causal mask with attention mask (both must allow attention)
            combined_mask = causal_mask & (attn_mask_block > 0)
            QK_block = QK_block + tl.where(combined_mask, 0, -1.0e6)
        else:
            # Only apply causal mask
            QK_block = QK_block + tl.where(causal_mask, 0, -1.0e6)
            
        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
        QK_block -= m_ij[:, None]
    else:
        # Non-causal case (blocks to the left of diagonal)
        QK_block = QK_block * softmax_scale
        
        # If attention mask is provided, apply it
        if attn_mask_ptr is not None:
            # Load the attention mask for the current block
            attn_mask_block = tl.load(attn_mask_ptr)
            # Apply attention mask
            QK_block = QK_block + tl.where(attn_mask_block > 0, 0, -1.0e6)
            
        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
        QK_block -= m_ij[:, None]

    P_block = tl.math.exp(QK_block)
    l_ij = tl.sum(P_block, 1)
    alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij

    V_block = tl.load(V_block_ptr)
    P_block = P_block.to(tl.float16)
    O_block = O_block * alpha[:, None]
    O_block = tl.dot(P_block, V_block, O_block)

    return O_block, l_i, m_ij


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in [3, 4, 7]
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def causal_flash_attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    attn_mask=None,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    qvk_offset = index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q into SRAM
    Q_block = tl.load(Q_block_ptr)

    lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    
    K_block_ptr_left = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr_left = tl.advance(V_block_ptr, (lo, 0))

    # Loop over k, v blocks to the left of the diagonal
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        attn_mask_ptr_left = None
        if attn_mask is not None:
            attn_mask_ptr_left = tl.make_block_ptr(
                base=attn_mask + qvk_offset,
                shape=(SEQ_LEN, SEQ_LEN),
                strides=(stride_Q_seq, stride_Q_seq),
                offsets=(offs_q[0], start_kv),
                block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KV),
                order=(1, 0),
            )
        
        O_block, l_i, m_i = causal_flash_attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr_left,
            V_block_ptr_left,
            start_kv,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            offs_q,
            offs_kv,
            False,  # non-causal (left of diagonal)
            attn_mask_ptr=attn_mask_ptr_left,
        )

        V_block_ptr_left = tl.advance(V_block_ptr_left, (BLOCK_SIZE_KV, 0))
        K_block_ptr_left = tl.advance(K_block_ptr_left, (0, BLOCK_SIZE_KV))


    # Process the diagonal block with causal masking
    lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
    lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    
    K_block_ptr_diag = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr_diag = tl.advance(V_block_ptr, (lo, 0))
    
    attn_mask_ptr_diag = None
    if attn_mask is not None:
        attn_mask_ptr_diag = tl.make_block_ptr(
            base=attn_mask + qvk_offset,
            shape=(SEQ_LEN, SEQ_LEN),
            strides=(stride_Q_seq, stride_Q_seq),
            offsets=(offs_q[0], lo),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KV),
            order=(1, 0),
        )
    
    O_block, l_i, m_i = causal_flash_attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr_diag,
        V_block_ptr_diag,
        lo,
        softmax_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        offs_q,
        offs_kv,
        True,  # causal (diagonal block)
        attn_mask_ptr=attn_mask_ptr_diag,
    )

    O_block = O_block / l_i[:, None]
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


class TritonCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale=None, attention_mask=None):
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        if softmax_scale is None:
            softmax_scale = 1.0 / (HEAD_DIM ** 0.5)
            
        O = torch.empty_like(Q)

        # Process attention mask if provided
        # Expected shape: [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
        if attention_mask is not None:
            # Expand attention mask to match the expected shape
            if attention_mask.dim() == 4:
                if attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1:
                    # Shape [batch_size, 1, 1, seq_len] -> expand to [batch_size, num_heads, seq_len, seq_len]
                    attention_mask = attention_mask.expand(-1, NUM_HEADS, SEQ_LEN, -1)
                elif attention_mask.shape[1] == 1:
                    # Shape [batch_size, 1, seq_len, seq_len] -> expand to [batch_size, num_heads, seq_len, seq_len]
                    attention_mask = attention_mask.expand(-1, NUM_HEADS, -1, -1)
            else:
                raise ValueError(f"Attention mask must be 4D with shape [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len], got {attention_mask.shape}")
            
            # Make sure the mask is contiguous for efficient memory access in the kernel
            attention_mask = attention_mask.contiguous()
        
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        causal_flash_attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM,
            attn_mask=attention_mask,
        )

        return O


def run_vanilla_attn(Q, K, V):
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

    attn_scores = torch.matmul(Q, K.transpose(-1, -2)) * softmax_scale
    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=Q.device), diagonal=1)
    attn_scores = attn_scores.masked_fill(mask.bool().unsqueeze(0).unsqueeze(0), float('-inf'))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, V)


def run_flash_attn(Q, K, V):
    softmax_scale = 1.0 / (Q.shape[-1] ** 0.5)
    return TritonCausalAttention.apply(Q, K, V, softmax_scale)


def benchmark_attn(batch_size, num_heads, seq_len, head_dim, num_runs=10):
    Q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16, device="cuda")
    K = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16, device="cuda")
    V = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=torch.float16, device="cuda")
    
    # warmup gpus
    with torch.no_grad():
        vanilla_output = run_vanilla_attn(Q, K, V)
        flash_output = run_flash_attn(Q, K, V)
    
    # verify outputs
    max_diff = torch.max(torch.abs(vanilla_output - flash_output))
    print(f"Max difference between vanilla and flash attention: {max_diff}")
    
    # benchmark vanilla attention
    torch.cuda.synchronize()
    vanilla_start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            run_vanilla_attn(Q, K, V)
        torch.cuda.synchronize()
    vanilla_end = time.time()
    vanilla_time = (vanilla_end - vanilla_start) / num_runs
    
    # benchmark flash attention
    torch.cuda.synchronize()
    flash_start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            run_flash_attn(Q, K, V)
        torch.cuda.synchronize()
    flash_end = time.time()
    flash_time = (flash_end - flash_start) / num_runs
    
    print(f"Vanilla attention: {vanilla_time * 1000:.3f} ms")
    print(f"Flash attention: {flash_time * 1000:.3f} ms")
    print(f"Speedup: {vanilla_time / flash_time:.2f}x")
    
    return {
        'vanilla_time': vanilla_time,
        'flash_time': flash_time,
        'speedup': vanilla_time / flash_time,
        'max_diff': max_diff.item()
    }


if __name__ == "__main__":
    print("Testing causal flash attention")
    for batch_size, num_heads, seq_len, head_dim in [
        (2, 4, 1024, 64),
        (4, 8, 2048, 64),
        (8, 16, 4096, 64),
    ]:
        print(f"\nRunning benchmark with BATCH_SIZE={batch_size}, NUM_HEADS={num_heads}, SEQ_LEN={seq_len}, HEAD_DIM={head_dim}")
        benchmark_attn(batch_size, num_heads, seq_len, head_dim)