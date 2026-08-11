import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _inverse_softplus(x: torch.Tensor):
    """Numerically stable inverse of softplus for positive initializers."""
    return x + torch.log(-torch.expm1(-x))


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

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        return x_dequant


class DDCLQuantizer(nn.Module):
    """Bounded DDCL channel that remains exactly packable as fixed-width INT.

    Each group stores an unconstrained latent ``h`` in the parent QuantLinear's
    weight parameter.  It is mapped to a bounded weight using

        z = alpha * tanh(h) - delta / 2

    where ``rho = alpha / delta`` is learned but constrained to
    ``rho <= 2**(bits-1) - 0.5``.  The half-step shift makes the bounded interval
    match the asymmetric signed codes ``[-2**(b-1), 2**(b-1)-1]`` used by the
    existing integer-zero-point backend.

    Training uses subtractive dither and the DDCL identity-through-detach path.
    Evaluation uses the same grid with zero dither.  The range construction
    guarantees that every code fits the final fixed-bit representation; clamps
    remain only as numerical guards.
    """

    def __init__(self, n_bits=8, group_size=None, weight=None):
        nn.Module.__init__(self)
        assert weight is not None, "DDCL initialization requires pretrained weights"
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** n_bits - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % self.group_size == 0
        self.enable = True

        self.signed_qmin = -(2 ** (n_bits - 1))
        self.signed_qmax = 2 ** (n_bits - 1) - 1
        self.rho_max = 2 ** (n_bits - 1) - 0.5
        self._rho_init_fraction = 0.95
        self._latent_margin = 0.95

        grouped = weight.detach().float().reshape(-1, self.group_size)
        rho_init = self.rho_max * self._rho_init_fraction
        center_fraction = 0.5 / rho_init
        group_max = grouped.amax(dim=-1, keepdim=True)
        group_min = grouped.amin(dim=-1, keepdim=True)
        positive_denom = max(self._latent_margin - center_fraction, 1e-3)
        negative_denom = max(self._latent_margin + center_fraction, 1e-3)
        alpha = torch.maximum(
            group_max.clamp_min(0.0) / positive_denom,
            (-group_min).clamp_min(0.0) / negative_denom,
        ).clamp_min(CLIPMIN)

        self.raw_alpha = nn.Parameter(_inverse_softplus(alpha).to(weight.dtype))
        rho_logit = math.log(self._rho_init_fraction / (1.0 - self._rho_init_fraction))
        self.raw_rho = nn.Parameter(torch.full_like(alpha, rho_logit).to(weight.dtype))
        self.register_buffer(
            "zero_point",
            torch.full_like(alpha, 2 ** (n_bits - 1)).to(weight.dtype),
        )

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha).clamp(min=CLIPMIN, max=1e4)

    @property
    def rho(self):
        return self.rho_max * torch.sigmoid(self.raw_rho)

    @property
    def scale(self):
        return (self.alpha / self.rho.clamp_min(1e-6)).clamp(min=CLIPMIN, max=1e4)

    def initial_latent(self, weight):
        """Return h such that bounded_weight(h) reconstructs ``weight``."""
        grouped = weight.detach().float().reshape(-1, self.group_size)
        normalized = (grouped + 0.5 * self.scale) / self.alpha
        normalized = normalized.clamp(-self._latent_margin, self._latent_margin)
        return torch.atanh(normalized).reshape_as(weight).to(weight.dtype)

    def bounded_weight(self, latent):
        dim1, dim2 = latent.shape
        grouped = latent.reshape(-1, self.group_size)
        bounded = self.alpha * torch.tanh(grouped) - 0.5 * self.scale
        return bounded.reshape(dim1, dim2)

    def _normalized_code(self, latent, dither):
        z = self.bounded_weight(latent).reshape(-1, self.group_size)
        normalized = z / self.scale
        signed_code = round_ste(normalized + dither)
        # The alpha/rho parameterization already guarantees these bounds.  This
        # clamp protects only against finite-precision boundary roundoff.
        signed_code = signed_code.clamp(self.signed_qmin, self.signed_qmax)
        return z, signed_code

    def fake_quant(self, latent):
        dim1, dim2 = latent.shape
        if self.training:
            dither = torch.rand_like(latent.reshape(-1, self.group_size)) - 0.5
        else:
            dither = torch.zeros_like(latent.reshape(-1, self.group_size))

        z, signed_code = self._normalized_code(latent, dither)
        z_hat = self.scale * (signed_code - dither)
        z_approx = z + (z_hat - z).detach()
        return z_approx.reshape(dim1, dim2)

    def ddcl_bit_cost(self, latent):
        """DDCL differentiable rate surrogate log2(|z| / delta + 1)."""
        z = self.bounded_weight(latent).reshape(-1, self.group_size)
        return torch.log1p((z / self.scale).abs()).div(math.log(2.0)).mean()

    def code_range_violation(self, latent):
        """Fraction of codes outside the packable range before the safety clamp."""
        z = self.bounded_weight(latent).reshape(-1, self.group_size)
        raw_code = torch.round(z / self.scale)
        invalid = (raw_code < self.signed_qmin) | (raw_code > self.signed_qmax)
        return invalid.float().mean()

    def rho_utilization(self):
        """Mean fraction of the available fixed-bit code radius in use."""
        return (self.rho / self.rho_max).mean()

    def forward(self, latent):
        if self.n_bits >= 16 or not self.enable:
            return self.bounded_weight(latent)
        return self.fake_quant(latent)


def get_quantizer(scheme: str, n_bits=8, group_size=None, weight=None):
    if scheme == "uniform_affine":
        return UniformAffineQuantizer(n_bits, group_size, weight=weight)
    elif scheme == "ddcl":
        return DDCLQuantizer(n_bits, group_size, weight=weight)
    else:
        raise ValueError(f"Unknown quantizer scheme: {scheme}")
