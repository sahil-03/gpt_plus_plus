from extensions.flash_attn_kernels import TritonCausalAttention
import torch
from torch import nn


class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Calculate softmax scale based on hidden_size and num_attention_heads
        # This follows the standard practice of scaling by 1/sqrt(head_dim)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.softmax_scale = 1.0 / (self.head_dim ** 0.5)
        
        # Store dropout probability for potential future use
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(self, query, key, value, attention_mask=None):
        """
        Flash attention implementation using Triton kernels
        
        Args:
            query: [batch_size, num_heads, seq_len, head_dim]
            key: [batch_size, num_heads, seq_len, head_dim]
            value: [batch_size, num_heads, seq_len, head_dim]
            attention_mask: Optional attention mask [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
                            where 1 indicates tokens to attend to and 0 indicates tokens to ignore
            
        Returns:
            output: [batch_size, num_heads, seq_len, head_dim]
        """
        # Process attention mask if provided
        if attention_mask is not None:
            # Make sure attention_mask is properly shaped
            if attention_mask.dim() == 3:
                # Convert [batch_size, 1, seq_len] to [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1)
            
            # Ensure mask values are binary (0 or 1)
            if attention_mask.dtype != torch.bool:
                # Convert to binary mask where 1 means "attend" and 0 means "ignore"
                attention_mask = attention_mask > 0
            
            # Convert to float for the kernel
            attention_mask = attention_mask.to(dtype=query.dtype)
        
        # Pass the query, key, value, softmax_scale, and attention_mask to the kernel
        # The kernel now properly handles the attention mask
        return TritonCausalAttention.apply(query, key, value, self.softmax_scale, attention_mask)
