import torch
import torch.nn as nn
import torch.nn.functional as F

class Quadapter(nn.Module):
    def __init__(self, hidden_size, reduction_factor=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.reduced_dim = hidden_size // reduction_factor

        # Down projection
        self.down_proj = nn.Linear(hidden_size, self.reduced_dim)
        
        # Quantization-aware parameters
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1, dtype=torch.int8))  # Actually use int8
        
        # Up projection
        self.up_proj = nn.Linear(self.reduced_dim, hidden_size)
        
        # Layer norm for stabilizing training
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # For calibration during training
        self.calibration = True
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def quantize(self, x):
        # Simulated quantization
        x_scaled = x / self.scale
        x_clipped = torch.clamp(x_scaled + self.zero_point, 0, 255)
        x_quantized = torch.round(x_clipped)
        x_dequantized = (x_quantized - self.zero_point) * self.scale
        return x_dequantized
    
    def forward(self, x):
        # Down projection
        down = self.down_proj(x)
        
        # Apply activation
        down = F.gelu(down)
        
        # Quantize the reduced representation
        quantized = self.quantize(down)
        
        # Up projection
        up = self.up_proj(quantized)
        
        # Layer norm and residual connection
        output = self.layer_norm(x + up)
        
        if self.calibration:
            with torch.no_grad():
                self.min_val = min(self.min_val, x.min().item())
                self.max_val = max(self.max_val, x.max().item())
        
        if self.training:
            # Need gradients during training
            return (quantized.float() * self.scale)
        else:
            # Keep as int8 during inference
            return quantized
        
        return output 

    def prepare_for_inference(self):
        self.calibration = False
        # Quantize the weights themselves
        self.down_proj.weight.data = self.quantize(self.down_proj.weight.data) 