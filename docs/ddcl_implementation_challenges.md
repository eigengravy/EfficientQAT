# Bounded DDCL in EfficientQAT

## Scope

This repository compares two Block-AP quantizers under the same fixed-width
deployment budget:

- `uniform_affine`: the standard EfficientQAT affine quantizer.
- `ddcl`: a bounded subtractive-dither channel with a differentiable DDCL rate
  surrogate.

Both produce the same INT2/INT3/INT4 packed format and use the same E2E-QP
phase. DDCL therefore tests whether its Block-AP training channel improves
quality at equal storage; it does not claim variable-length checkpoint savings.

## Bounded latent

For each weight group, the pretrained weight is represented by a latent `h`:

```text
rho   = rho_max * sigmoid(raw_rho)
delta = alpha / rho
z     = alpha * tanh(h) - delta / 2
```

`alpha` is positive through a softplus parameterization. The initial `h` is
chosen with `atanh` so that `z` reconstructs the pretrained weights before
quantization.

For `b` bits:

```text
rho_max = 2^(b-1) - 1/2
```

The `-delta/2` affine center aligns the bounded interval with the signed integer
codes supported by EfficientQAT's integer-zero-point format:

```text
signed codes: [-2^(b-1), 2^(b-1)-1]
stored codes: [0, 2^b-1]
zero point:   2^(b-1)
```

This construction guarantees that DDCL cannot learn a resolution requiring
more codes than the final model can store.

## Training and evaluation channels

Training uses normalized subtractive dither:

```text
e       ~ Uniform(-1/2, 1/2)
m       = round(z / delta + e)
z_hat   = delta * (m - e)
z_approx = z + stop_gradient(z_hat - z)
```

Evaluation and final materialization set `e = 0`. `round` is used instead of
the algebraically shifted `floor`/bin-center form because it maps exactly to the
existing integer-zero-point backend. It retains the same subtractive-dither
error channel without requiring a fractional zero point.

The final clamp is a numerical guard, not part of the optimization strategy.
`block_ap/ddcl_code_range_violation_mean` should remain exactly zero.

## Rate objective

Block-AP adds:

```text
lambda * mean(log2(|z| / delta + 1))
```

The rate term can reduce the effective number of occupied codes by pulling
unnecessary coordinates toward the zero code and by learning a smaller `rho`.
Because storage remains fixed-width, report it as a rate surrogate rather than
physical bits saved.

Useful metrics are:

```text
block_ap/val_loss_mean
block_ap/ddcl_bit_cost_mean
block_ap/ddcl_rho_utilization_mean
block_ap/ddcl_code_range_violation_mean
```

Checkpoint selection uses deterministic validation loss, and Block-AP restores
the best epoch before materialization and packing.
