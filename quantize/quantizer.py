import torch
import torch.nn as nn
import math

import pdb

CLIPMIN = 1e-4



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def round_dithered_ste(x: torch.Tensor):
    """Subtractive-dither quantization with identity backward.

    Forward implements the normalized DDCL channel:
        q = round(x + eps), eps ~ U(-0.5, 0.5)
        x_hat = q - eps

    The identity backward is then the pathwise gradient for the dithered
    reconstruction in the unclipped region, rather than plain stochastic
    rounding's biased STE surrogate.
    """
    eps = torch.rand_like(x) - 0.5
    dithered = (x + eps).round() - eps
    return (dithered - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True

        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1,self.group_size)
                xmin = x.amin([-1], keepdim=True)
                xmax =  x.amax([-1], keepdim=True)
                range = xmax - xmin
                scale = range / (2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4)
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())


    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)

    def fake_quant(self, x):
        scale = clamp_ste(self.scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant

    def ddcl_bit_cost(self, x):
        """Differentiable DDCL-style code-length upper bound per weight.

        This is not used by the baseline quantizer unless the training loop adds
        it to the loss. The bound follows log2(2 * |z| / delta + 1), with the
        learned quantizer scale acting as delta.
        """
        delta = clamp_ste(self.scale, 1e-4, 1e4)
        x = x.reshape(-1, self.group_size)
        normalized_magnitude = (x / delta).abs()
        return torch.log1p(2.0 * normalized_magnitude).div(math.log(2.0)).mean()

    def ddcl_saturation_rate(self, x):
        """Expected fixed-range overflow rate under subtractive dither.

        DDCL training can use an unclamped integer channel, but the final model
        is still projected into [qmin, qmax]. This metric estimates the
        probability that round(x / scale + eps) + zero_point would fall outside
        that fixed range for eps ~ U(-0.5, 0.5).
        """
        delta = clamp_ste(self.scale, 1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        x = x.reshape(-1, self.group_size)
        normalized = x / delta
        lower = self.qmin - round_zero_point
        upper = self.qmax - round_zero_point
        lower_overflow = (lower - normalized).clamp(min=0.0, max=1.0)
        upper_overflow = (normalized - upper).clamp(min=0.0, max=1.0)
        return (lower_overflow + upper_overflow).mean()


    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        return x_dequant


class DDCLQuantizer(UniformAffineQuantizer):
    """DDCL-inspired quantizer.

    Training uses subtractive-dither fake quantization plus an optional bit-cost
    term added by the Block-AP loop. Eval/final packing still uses deterministic
    rounding so the saved model remains compatible with fixed-width kernels.
    """

    def fake_quant(self, x):
        scale = clamp_ste(self.scale, 1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        if self.training:
            eps = torch.rand_like(x) - 0.5
            x_int = round_ste(x / scale + eps)
        else:
            eps = None
            x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        if not self.training:
            x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        if eps is not None:
            x_dequant = x_dequant.sub(eps)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant


def get_quantizer(scheme: str, n_bits=8, group_size=None, weight=None):
    if scheme == "uniform_affine":
        return UniformAffineQuantizer(n_bits, group_size, weight=weight)
    elif scheme == "ddcl":
        return DDCLQuantizer(n_bits, group_size, weight=weight)
    else:
        raise ValueError(f"Unknown quantizer scheme: {scheme}")
