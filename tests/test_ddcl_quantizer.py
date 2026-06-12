"""Tests for DDCLQuantizer and round_dithered_ste."""
import sys
sys.path.insert(0, sys.path[0] + '/..')

import pytest
import torch
from quantize.quantizer import (
    round_ste, round_dithered_ste,
    UniformAffineQuantizer, DDCLQuantizer, get_quantizer,
)


def test_round_dithered_ste_is_stochastic():
    x = torch.tensor([2.3, 4.7, 1.5])
    results = [round_dithered_ste(x) for _ in range(100)]
    stacked = torch.stack(results)
    assert not torch.all(stacked == stacked[0]), "Should produce varying outputs"


def test_round_dithered_ste_unbiased():
    torch.manual_seed(42)
    x = torch.tensor([2.3, 4.7, 1.5, 0.1, 3.9])
    n_samples = 50000
    results = torch.stack([round_dithered_ste(x) for _ in range(n_samples)])
    mean = results.mean(dim=0)
    # Unbiased: E[round_dithered_ste(x)] ≈ x
    assert torch.allclose(mean, x, atol=0.05), f"Expected mean ≈ {x}, got {mean}"


def test_round_dithered_ste_is_subtractive_not_integer_rounding():
    torch.manual_seed(0)
    x = torch.full((1000,), 2.3)
    y = round_dithered_ste(x)
    residual = y - x
    # True subtractive dither reconstructs q - eps, so the result should not
    # be locked to integer grid points during training.
    assert not torch.allclose(y, y.round())
    assert residual.min().item() >= -0.5001
    assert residual.max().item() <= 0.5001
    assert abs(residual.mean().item()) < 0.05


def test_round_dithered_ste_gradient():
    x = torch.tensor([2.3, 4.7], requires_grad=True)
    y = round_dithered_ste(x)
    loss = y.sum()
    loss.backward()
    # Subtractive dither uses identity backward for the pathwise gradient.
    assert torch.allclose(x.grad, torch.ones_like(x)), f"Grad should be 1, got {x.grad}"


def test_round_ste_gradient():
    x = torch.tensor([2.3, 4.7], requires_grad=True)
    y = round_ste(x)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x.grad, torch.ones_like(x)), f"Grad should be 1, got {x.grad}"


def test_ddcl_quantizer_inherits_uniform():
    assert issubclass(DDCLQuantizer, UniformAffineQuantizer)


def test_ddcl_quantizer_eval_is_deterministic():
    weight = torch.randn(64, 128)
    q = DDCLQuantizer(n_bits=4, group_size=128, weight=weight)
    q.eval()
    out1 = q(weight)
    out2 = q(weight)
    assert torch.equal(out1, out2), "Eval mode should be deterministic"


def test_ddcl_quantizer_train_is_stochastic():
    weight = torch.randn(64, 128)
    q = DDCLQuantizer(n_bits=4, group_size=128, weight=weight)
    q.train()
    out1 = q(weight)
    out2 = q(weight)
    assert not torch.equal(out1, out2), "Train mode should be stochastic"


def test_ddcl_quantizer_train_uses_subtractive_dither():
    weight = torch.tensor([[2.3, -1.2, 0.4, 3.7]], dtype=torch.float32)
    q = DDCLQuantizer(n_bits=8, group_size=4, weight=weight)
    with torch.no_grad():
        q.scale.fill_(1.0)
        q.zero_point.fill_(16.0)
    q.train()
    torch.manual_seed(123)
    out = q(weight)
    residual = out - weight
    assert not torch.allclose(out, out.round())
    assert residual.min().item() >= -0.5001
    assert residual.max().item() <= 0.5001


def test_ddcl_train_is_unclamped_but_eval_is_clamped():
    weight = torch.tensor([[10.0]], dtype=torch.float32)
    q = DDCLQuantizer(n_bits=2, group_size=1, weight=weight)
    with torch.no_grad():
        q.scale.fill_(1.0)
        q.zero_point.fill_(0.0)

    q.train()
    torch.manual_seed(0)
    train_out = q(weight)
    assert train_out.item() > q.qmax

    q.eval()
    eval_out = q(weight)
    assert eval_out.item() == float(q.qmax)


def test_ddcl_saturation_rate_detects_fixed_range_overflow():
    weight = torch.tensor([[10.0, 2.0]], dtype=torch.float32)
    q = DDCLQuantizer(n_bits=2, group_size=2, weight=weight)
    with torch.no_grad():
        q.scale.fill_(1.0)
        q.zero_point.fill_(0.0)

    rate = q.ddcl_saturation_rate(weight)
    assert 0.49 < rate.item() < 0.51


def test_uniform_quantizer_always_deterministic():
    weight = torch.randn(64, 128)
    q = UniformAffineQuantizer(n_bits=4, group_size=128, weight=weight)
    q.train()
    out1 = q(weight)
    out2 = q(weight)
    assert torch.equal(out1, out2), "Uniform quantizer should always be deterministic"


def test_get_quantizer_factory():
    weight = torch.randn(64, 128)
    q_uniform = get_quantizer("uniform_affine", n_bits=4, group_size=128, weight=weight)
    q_ddcl = get_quantizer("ddcl", n_bits=4, group_size=128, weight=weight)
    assert isinstance(q_uniform, UniformAffineQuantizer)
    assert isinstance(q_ddcl, DDCLQuantizer)


def test_get_quantizer_unknown_scheme_raises():
    weight = torch.randn(64, 128)
    with pytest.raises(ValueError, match="Unknown quantizer scheme"):
        get_quantizer("baseline", n_bits=4, group_size=128, weight=weight)


def test_ddcl_output_in_valid_range():
    weight = torch.randn(64, 128) * 3
    q = DDCLQuantizer(n_bits=2, group_size=128, weight=weight)
    q.train()
    for _ in range(10):
        out = q(weight)
        # Dequantized output should be finite
        assert torch.isfinite(out).all()


def test_ddcl_gradient_flows_through_quantizer():
    weight = torch.randn(64, 128, requires_grad=True)
    q = DDCLQuantizer(n_bits=4, group_size=128, weight=weight.detach())
    q.train()
    # Simulate forward pass
    out = q(weight)
    loss = out.sum()
    loss.backward()
    assert weight.grad is not None
    assert weight.grad.abs().sum() > 0, "Gradients should flow through quantizer"


def test_ddcl_bit_cost_is_positive_and_differentiable():
    weight = torch.randn(64, 128, requires_grad=True)
    q = DDCLQuantizer(n_bits=4, group_size=128, weight=weight.detach())
    bit_cost = q.ddcl_bit_cost(weight)
    bit_cost.backward()
    assert bit_cost.item() > 0
    assert weight.grad is not None
    assert weight.grad.abs().sum() > 0


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {type(e).__name__}: {e}")
    print(f"\nRan {len(tests)} tests")
