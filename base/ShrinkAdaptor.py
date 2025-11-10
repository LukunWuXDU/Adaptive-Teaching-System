import torch.nn as nn
import torch

# ShrinkAdaptor.py

class IdentityAdapter(nn.Module):
    """
    A "pseudo-Adapter" used to establish a clear baseline. It does not alter the input and enables the unaltered propagation of features.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class LinearAdapter(nn.Module):
    """Linear Adapter"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adapter = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.adapter(x)


class MLPAdapter(nn.Module):
    """Multi-Layer Perceptron (MLP) Adapter"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.adapter(x)





class DeepAdapter(nn.Module):
    """Multi-Layer Non-Linear Adapter"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.adapter(x)


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)        

class ImgProjection(nn.Module):
    """ImgProjection with ResidualAdd, GELU, Dropout, and LayerNorm"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.Dropout(0.3),
            )),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.adapter(x)


# Our ShrinkAdapter
class ShrinkAdapter(nn.Module):
    """
    Adapter with a bottleneck architecture and residual connection free.

    This adapter takes an input tensor, passes it through a 'bottleneck'
    (down-projection -> non-linearity -> up-projection) to refine the features.
    """
    def __init__(self, input_dim, output_dim, bottleneck_ratio=0.25, dropout_rate=0.3):
        """
        Args:
            input_dim (int): The feature dimension of the input and output.
            bottleneck_ratio (float): The ratio to determine the size of the bottleneck.
                                      e.g., 0.25 means the bottleneck dim will be 1/4 of input_dim.
            dropout_rate (float): The dropout rate.
        """
        super().__init__()

        self.input_dim = input_dim
        
        bottleneck_dim = int(input_dim * bottleneck_ratio)
        
        self.adapter = nn.Sequential(

            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),  
            nn.Linear(bottleneck_dim, output_dim),
            nn.Dropout(dropout_rate)  
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.shape[-1] == self.input_dim, "Input dimension mismatch"

        adapter_output = self.adapter(x)
        final_output = self.layer_norm(adapter_output)
        
        return final_output


