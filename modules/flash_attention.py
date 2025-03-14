from extensions.flash_attn_kernels import TritonCausalAttention
import torch
from torch import nn


class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.softmax_scale = 1.0 / (self.head_dim ** 0.5)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # Ensure hidden_states is float16 for better performance with flash attention
        hidden_states = hidden_states.to(torch.float16)
        
        # First, we have to generate the key, value, query for each token for multi-head attention
        # using self.transform (more details inside the function).
        # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        # Reshape hidden states to prepare for flash attention
        query = hidden_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = hidden_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value = hidden_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Keep attention_mask as is for indexing operations in the model
        # but create a float16 copy for the flash attention computation if needed
        attn_mask_for_flash = None
        if attention_mask is not None:
            attn_mask_for_flash = attention_mask.to(torch.float16)
        
        # Apply flash attention
        attn_output = TritonCausalAttention.apply(query, key, value, self.softmax_scale, attn_mask_for_flash)
        
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return attn_output
