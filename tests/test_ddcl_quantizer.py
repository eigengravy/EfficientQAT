"""Tests for DDCLFixedLengthQuantizer and round_dithered_ste."""
import sys
sys.path.insert(0, sys.path[0] + '/..')

import pytest
import torch
from quantize.quantizer import (
    round_ste, round_dithered_ste,
    UniformAffineQuantizer, DDCLFixedLengthQuantizer, get_quantizer,
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


def test_round_dithered_ste_probability():
    torch.manual_seed(0)
    x = torch.tensor([2.3])
    n_samples = 50000
    results = torch.stack([round_dithered_ste(x) for _ in range(n_samples)])
    # P(round up to 3) should be ≈ 0.3, P(round down to 2) ≈ 0.7
    p_up = (results == 3.0).float().mean().item()
    assert 0.25 < p_up < 0.35, f"P(round up) = {p_up}, expected ~0.3"


def test_round_dithered_ste_gradient():
    x = torch.tensor([2.3, 4.7], requires_grad=True)
    y = round_dithered_ste(x)
    loss = y.sum()
    loss.backward()
    # STE: gradient should be 1.0
    assert torch.allclose(x.grad, torch.ones_like(x)), f"Grad should be 1, got {x.grad}"


def test_round_ste_gradient():
    x = torch.tensor([2.3, 4.7], requires_grad=True)
    y = round_ste(x)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x.grad, torch.ones_like(x)), f"Grad should be 1, got {x.grad}"


def test_ddcl_quantizer_inherits_uniform():
    assert issubclass(DDCLFixedLengthQuantizer, UniformAffineQuantizer)


def test_ddcl_quantizer_eval_is_deterministic():
    weight = torch.randn(64, 128)
    q = DDCLFixedLengthQuantizer(n_bits=4, group_size=128, weight=weight)
    q.eval()
    out1 = q(weight)
    out2 = q(weight)
    assert torch.equal(out1, out2), "Eval mode should be deterministic"


def test_ddcl_quantizer_train_is_stochastic():
    weight = torch.randn(64, 128)
    q = DDCLFixedLengthQuantizer(n_bits=4, group_size=128, weight=weight)
    q.train()
    out1 = q(weight)
    out2 = q(weight)
    assert not torch.equal(out1, out2), "Train mode should be stochastic"


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
    q_ddcl = get_quantizer("ddcl_fixed", n_bits=4, group_size=128, weight=weight)
    assert isinstance(q_uniform, UniformAffineQuantizer)
    assert isinstance(q_ddcl, DDCLFixedLengthQuantizer)


def test_get_quantizer_unknown_scheme_raises():
    weight = torch.randn(64, 128)
    with pytest.raises(ValueError, match="Unknown quantizer scheme"):
        get_quantizer("baseline", n_bits=4, group_size=128, weight=weight)


def test_ddcl_output_in_valid_range():
    weight = torch.randn(64, 128) * 3
    q = DDCLFixedLengthQuantizer(n_bits=2, group_size=128, weight=weight)
    q.train()
    for _ in range(10):
        out = q(weight)
        # Dequantized output should be finite
        assert torch.isfinite(out).all()


def test_ddcl_gradient_flows_through_quantizer():
    weight = torch.randn(64, 128, requires_grad=True)
    q = DDCLFixedLengthQuantizer(n_bits=4, group_size=128, weight=weight.detach())
    q.train()
    # Simulate forward pass
    out = q(weight)
    loss = out.sum()
    loss.backward()
    assert weight.grad is not None
    assert weight.grad.abs().sum() > 0, "Gradients should flow through quantizer"


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
