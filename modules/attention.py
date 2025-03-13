import torch
from einops import rearrange
from torch import nn
import math


class FlashAttention(nn.Module):
    def __init__(self, dropout_prob=0.1, block_size=1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.block_size = block_size
        
    def _flash_attention_forward(self, q, k, v, mask=None):
        """
        Implements the core Flash Attention algorithm with tiled computation
        q, k, v: query, key, value tensors [b, h, t, d]
        mask: attention mask [b, 1, 1, t] or None
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Initialize output and attention statistics
        out = torch.zeros_like(q)
        softmax_scale = 1.0 / math.sqrt(head_dim)
        
        # For each query block
        for start_q in range(0, seq_len, self.block_size):
            end_q = min(start_q + self.block_size, seq_len)
            q_block = q[:, :, start_q:end_q, :]
            
            # Initialize block statistics
            block_m = torch.ones(batch_size, num_heads, end_q - start_q, 1, 
                                device=q.device) * float('-inf')
            block_l = torch.zeros(batch_size, num_heads, end_q - start_q, 1, 
                                 device=q.device)
            block_o = torch.zeros(batch_size, num_heads, end_q - start_q, head_dim, 
                                 device=q.device)
            
            # For each key-value block
            for start_k in range(0, seq_len, self.block_size):
                end_k = min(start_k + self.block_size, seq_len)
                k_block = k[:, :, start_k:end_k, :]
                v_block = v[:, :, start_k:end_k, :]
                
                # Compute attention scores for this block
                scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * softmax_scale
                
                # Apply mask if provided
                if mask is not None:
                    scores = scores + mask[:, :, :, start_k:end_k]
                
                # Handle causal masking (upper triangular part)
                if start_q > start_k:
                    # All tokens attend to all tokens in this block (no masking needed)
                    pass
                elif end_k <= start_q:
                    # No tokens attend to any tokens in this block
                    continue
                else:
                    # Causal masking within overlapping blocks
                    causal_mask = torch.triu(torch.ones(end_q - start_q, end_k - start_k, 
                                                      device=q.device), 
                                           diagonal=1 + start_k - start_q)
                    scores.masked_fill_(causal_mask.bool().unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # Calculate max for numerical stability
                block_m_new = torch.maximum(block_m, scores.max(dim=-1, keepdim=True)[0])
                
                # Update output and statistics
                exp_scores = torch.exp(scores - block_m_new)
                block_o_new = torch.matmul(exp_scores, v_block)
                
                block_l_new = block_l * torch.exp(block_m - block_m_new) + exp_scores.sum(dim=-1, keepdim=True)
                block_o = block_o * torch.exp(block_m - block_m_new) + block_o_new
                block_m = block_m_new
                block_l = block_l_new
                
            # Finalize output for this query block
            block_output = block_o / block_l
            out[:, :, start_q:end_q, :] = block_output
            
        return out

    def forward(self, query, key, value, attention_mask=None):
        # Apply dropout to the input tensors if needed during training
        if self.training and self.dropout.p > 0:
            query = self.dropout(query)
            key = self.dropout(key)
            value = self.dropout(value)
            
        # Execute the core Flash Attention algorithm
        output = self._flash_attention_forward(query, key, value, attention_mask)
        
        return output


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # This dropout is applied after attention
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Initialize Flash Attention with block size dependent on sequence length
        # Typically block size is chosen based on hardware constraints (L1/L2 cache sizes)
        self.flash_attention = FlashAttention(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # Project and reshape for multi-head attention
        proj = linear_layer(x)
        proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
        proj = rearrange(proj, 'b t h d -> b h t d')
        return proj

  def attention(self, key, query, value, attention_mask):
    # QK^T / sqrt(d_k)
    attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
    attn_scores = attn_scores + attention_mask
    # also add a causal mask to the attention scores
    causal_mask = torch.triu(torch.ones(attn_scores.size(-1), attn_scores.size(-1)), diagonal=1)
    causal_mask = causal_mask[None, None, :, :].to(attn_scores.device)
    attn_scores = attn_scores.masked_fill(causal_mask == 1, float('-inf'))
    # softmax per element in the sequence
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
    # Apply dropout to the attention probabilities.
    attn_probs = self.dropout(attn_probs)
    # apply the attention probabilities to the values
    attn_value = torch.matmul(attn_probs, value)
    attn_value = rearrange(attn_value, 'b h t d -> b t h d')
    attn_value = rearrange(attn_value, 'b t h d -> b t (h d)')
    return attn_value
    ### YOUR CODE HERE


def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
    # hidden_states: [bs, seq_len, hidden_state]
    # attention_mask: [bs, 1, 1, seq_len]
    # output: [bs, seq_len, hidden_state]
    # """
    # # First, we have to generate the key, value, query for each token for multi-head attention
    # # using self.transform (more details inside the function).
    # # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    # key_layer = self.transform(hidden_states, self.key)
    # value_layer = self.transform(hidden_states, self.value)
    # query_layer = self.transform(hidden_states, self.query)
    
    # # Calculate the multi-head attention.
    # attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    # return attn_value
    # hidden_states: [bs, seq_len, hidden_state]
    # attention_mask: [bs, 1, 1, seq_len]
    # output: [bs, seq_len, hidden_state]
    # """
    # # First, we have to generate the key, value, query for each token for multi-head attention
    # # using self.transform (more details inside the function).
    # # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    # key_layer = self.transform(hidden_states, self.key)
    # value_layer = self.transform(hidden_states, self.value)
    # query_layer = self.transform(hidden_states, self.query)
    
    # # Calculate the multi-head attention.
    # attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    # return attn_value
