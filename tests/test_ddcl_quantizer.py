"""Unit tests for the bounded, fixed-budget DDCL quantizer."""
import sys

sys.path.insert(0, sys.path[0] + "/..")

import pytest
import torch

from quantize.quantizer import (
    DDCLQuantizer,
    UniformAffineQuantizer,
    get_quantizer,
    round_dithered_ste,
    round_ste,
)


def make_ddcl(weight, bits=4, group_size=None):
    group_size = group_size or weight.shape[-1]
    quantizer = DDCLQuantizer(bits, group_size, weight=weight)
    latent = quantizer.initial_latent(weight)
    return quantizer, latent


def test_round_dithered_ste_is_unbiased_and_differentiable():
    torch.manual_seed(42)
    x = torch.tensor([2.3, 4.7, 1.5, 0.1], requires_grad=True)
    samples = torch.stack([round_dithered_ste(x) for _ in range(20000)])
    assert torch.allclose(samples.mean(dim=0), x, atol=0.05)
    samples[0].sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_round_ste_has_identity_backward():
    x = torch.tensor([2.3, 4.7], requires_grad=True)
    round_ste(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_bounded_parameterization_reconstructs_pretrained_weight():
    weight = torch.randn(32, 128)
    quantizer, latent = make_ddcl(weight)
    reconstructed = quantizer.bounded_weight(latent)
    assert torch.allclose(reconstructed, weight, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_learned_resolution_never_exceeds_fixed_bit_budget(bits):
    weight = torch.randn(16, 128)
    quantizer, _ = make_ddcl(weight, bits=bits)
    assert torch.all(quantizer.rho > 0)
    assert torch.all(quantizer.rho <= quantizer.rho_max)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_bounded_codes_fit_packable_signed_range(bits):
    weight = torch.randn(64, 128) * 4
    quantizer, latent = make_ddcl(weight, bits=bits)
    quantizer.eval()
    bounded = quantizer.bounded_weight(latent).reshape(-1, quantizer.group_size)
    signed_codes = torch.round(bounded / quantizer.scale)
    assert signed_codes.min().item() >= quantizer.signed_qmin
    assert signed_codes.max().item() <= quantizer.signed_qmax
    assert quantizer.code_range_violation(latent).item() == 0.0


def test_eval_grid_maps_exactly_to_integer_zero_point_backend():
    weight = torch.randn(32, 128)
    quantizer, latent = make_ddcl(weight, bits=4)
    quantizer.eval()
    output = quantizer(latent).reshape(-1, quantizer.group_size)
    unsigned_codes = output / quantizer.scale + quantizer.zero_point
    assert torch.allclose(unsigned_codes, unsigned_codes.round(), atol=1e-5)
    assert unsigned_codes.min().item() >= quantizer.qmin
    assert unsigned_codes.max().item() <= quantizer.qmax


def test_eval_is_deterministic_and_training_is_stochastic():
    weight = torch.randn(64, 128)
    quantizer, latent = make_ddcl(weight)
    quantizer.eval()
    assert torch.equal(quantizer(latent), quantizer(latent))
    quantizer.train()
    assert not torch.equal(quantizer(latent), quantizer(latent))


def test_training_uses_subtractive_dither():
    weight = torch.randn(64, 128)
    quantizer, latent = make_ddcl(weight)
    quantizer.train()
    output = quantizer(latent).reshape(-1, quantizer.group_size)
    normalized_output = output / quantizer.scale
    assert not torch.allclose(normalized_output, normalized_output.round())


def test_ddcl_identity_path_updates_latent():
    weight = torch.randn(64, 128)
    quantizer, latent = make_ddcl(weight)
    latent.requires_grad_(True)
    quantizer.train()
    quantizer(latent).sum().backward()
    assert latent.grad is not None
    assert latent.grad.abs().sum() > 0


def test_rate_surrogate_updates_latent_and_resolution_parameters():
    weight = torch.randn(64, 128)
    quantizer, latent = make_ddcl(weight)
    latent.requires_grad_(True)
    cost = quantizer.ddcl_bit_cost(latent)
    cost.backward()
    assert cost.item() > 0
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert quantizer.raw_alpha.grad is not None
    assert quantizer.raw_rho.grad is not None


def test_uniform_quantizer_behavior_is_unchanged():
    weight = torch.randn(64, 128)
    quantizer = UniformAffineQuantizer(4, 128, weight=weight)
    quantizer.train()
    assert torch.equal(quantizer(weight), quantizer(weight))


def test_quantizer_factory():
    weight = torch.randn(64, 128)
    assert isinstance(get_quantizer("uniform_affine", 4, 128, weight), UniformAffineQuantizer)
    assert isinstance(get_quantizer("ddcl", 4, 128, weight), DDCLQuantizer)
    with pytest.raises(ValueError, match="Unknown quantizer scheme"):
        get_quantizer("baseline", 4, 128, weight)
