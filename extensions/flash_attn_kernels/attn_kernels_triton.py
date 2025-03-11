@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def flash_attn_fwd_causal_kernel(
    Q,  # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    K,  # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    V,  # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    softmax_scale,
    M,  # [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
    O,  # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
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
):
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
        strides=(stride_K_dim, stride_K_seq),  # Transposed relative to Q.
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
    m_i = tl.full([BLOCK_SIZE_Q], float("-inf"), tl.float32)
    l_i = tl.full([BLOCK_SIZE_Q], 1.0, tl.float32)
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # Load Q block into SRAM.
    Q_block = tl.load(Q_block_ptr)

    # For causal attention we always use two inner stages:
    #   inner_stage 1: Process keys strictly before the current query block.
    #   inner_stage 2: Process the "transition" block with masking.
    inner_stages = [1, 2]

    orig_K_block_ptr = K_block_ptr
    orig_V_block_ptr = V_block_ptr

    for inner_stage in inner_stages:
        if inner_stage == 1:
            lo = 0
            hi = block_index_q * BLOCK_SIZE_Q
        elif inner_stage == 2:
            lo = block_index_q * BLOCK_SIZE_Q
            hi = (block_index_q + 1) * BLOCK_SIZE_Q
            lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
        else:
            raise ValueError("Unexpected inner stage in causal attention kernel")

        cur_K_block_ptr = tl.advance(orig_K_block_ptr, (0, lo))
        cur_V_block_ptr = tl.advance(orig_V_block_ptr, (lo, 0))

        for start_kv in range(lo, hi, BLOCK_SIZE_KV):
            start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
            K_block = tl.load(cur_K_block_ptr)
            QK_block = tl.dot(Q_block, K_block)

            if inner_stage == 2:
                mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
                QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
                QK_block = QK_block - m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
                QK_block = QK_block * softmax_scale - m_ij[:, None]

            P_block = tl.math.exp(QK_block)
            l_ij = tl.sum(P_block, 1)
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            V_block = tl.load(cur_V_block_ptr)
            P_block = P_block.to(tl.float16)
            O_block = O_block * alpha[:, None]
            O_block = tl.dot(P_block, V_block, O_block)

            m_i = m_ij

            # advance pointers
            cur_V_block_ptr = tl.advance(cur_V_block_ptr, (BLOCK_SIZE_KV, 0))
            cur_K_block_ptr = tl.advance(cur_K_block_ptr, (0, BLOCK_SIZE_KV))

    # finalize output
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))
