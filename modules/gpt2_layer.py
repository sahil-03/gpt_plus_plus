from torch import nn

import torch.nn.functional as F

from attention import CausalSelfAttention
from quadapter import Quadapter

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # Quadapter modules
    self.attention_quadapter = Quadapter(config.hidden_size)
    self.feed_forward_quadapter = Quadapter(config.hidden_size)
    
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def prepare_for_quantized_inference(self):
    """Prepare the layer for quantized inference by quantizing all Quadapter modules."""
    self.attention_quadapter.prepare_for_inference()
    self.feed_forward_quadapter.prepare_for_inference()
    
  @staticmethod
  def prepare_model_for_quantized_inference(model):
    """
    Prepare the entire model for quantized inference.
    Should be called after training and before saving for inference.
    
    Usage:
        # During training
        model.train()
        # ... train normally ...
        
        # After training, prepare for inference
        model.eval()
        GPT2Layer.prepare_model_for_quantized_inference(model)
        
        # Save model
        torch.save(model.state_dict(), 'quantized_model.pt')
    """
    for module in model.modules():
      if isinstance(module, GPT2Layer):
        module.prepare_for_quantized_inference()

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    ### YOUR CODE HERE
    dense_output = dense_layer(output)
    dropout_output = dropout(dense_output)
    return input + dropout_output

  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """

    ### YOUR CODE HERE
    hidden_norm = self.attention_layer_norm(hidden_states)
    # Apply Quadapter after the first layer norm
    hidden_norm = self.attention_quadapter(hidden_norm)
    masked_multi_attn = self.self_attention.forward(hidden_norm, attention_mask)
    interim = self.add(hidden_states, masked_multi_attn, self.attention_dense, self.attention_dropout)

    interim_norm = self.out_layer_norm(interim)
    # Apply Quadapter after the second layer norm
    interim_norm = self.feed_forward_quadapter(interim_norm)
    output = self.interm_af(self.interm_dense(interim_norm))
    return self.add(interim, output, self.out_dense, self.out_dropout)    
