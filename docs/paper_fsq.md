# FSQ: Finite Scalar Quantization — VQ-VAE Made Simple

**Paper**: Finite Scalar Quantization: VQ-VAE Made Simple  
**Authors**: Fabian Mentzer, David Minnen, Eirikur Agustsson, Michael Tschannen (Google Research)  
**Year**: 2023 (ICLR 2024)  
**arXiv**: [2309.15505](https://arxiv.org/abs/2309.15505)  
**Code**: [google-research/fsq](https://github.com/google-research/google-research/tree/master/fsq)

---

## 1. Core Idea

FSQ replaces VQ's learned codebook lookup with **independent per-scalar rounding** to a finite set of integer values. The "codebook" is implicit — it's the Cartesian product of per-dimension quantization levels. This eliminates codebook collapse, auxiliary losses, and complex training machinery while matching VQ performance.

---

## 2. The FSQ Quantization Function

For a d-dimensional representation z ∈ ℝ^d (typically d < 10):

### Step 1: Bound each scalar to a finite range
```
f(z_i) = ⌊L_i/2⌋ · tanh(z_i)
```

More precisely, the bounding function handles odd/even levels:
```python
def bound(z, L_i, eps=1e-3):
    half_l = (L_i - 1) * (1 - eps) / 2
    offset = 0.5 if (L_i % 2 == 0) else 0.0
    shift = arctanh(offset / half_l)
    return tanh(z + shift) * half_l - offset
```

### Step 2: Round to nearest integer
```
ẑ_i = round(f(z_i))
```

### Result
Each quantized scalar: `ẑ_i ∈ {-⌊L_i/2⌋, ..., ⌊L_i/2⌋}` (L_i possible values)

---

## 3. Implicit Codebook

The effective codebook is the Cartesian product of per-dimension value sets:

```
|C| = ∏(i=1 to d) L_i
```

Each codeword is a unique d-dimensional integer vector. The mapping from codeword to index is:
```
index = (ẑ · basis).sum()
where basis = cumprod(L[:-1])    (mixed-radix encoding)
```

**Example**: d=3, L=[8,6,5] → |C| = 8×6×5 = 240 ≈ 2^8

---

## 4. Straight-Through Estimator

Gradients flow through rounding via STE:

```
round_ste(x) = x + sg(round(x) - x)
```

Where `sg` = stop_gradient. Forward: rounds to integer. Backward: gradient magnitude 1 (identity).

The full quantization with STE:
```
ẑ = round_ste(bound(z))
```

---

## 5. Recommended Level Configurations

| Target |C| (bits) | Levels L | d | Actual |C| |
|--------|----------|---|---------|
| 2^8 (256) | [8, 6, 5] | 3 | 240 |
| 2^10 (1024) | [8, 5, 5, 5] | 4 | 1000 |
| 2^12 (4096) | [7, 5, 5, 5, 5] | 5 | 4375 |
| 2^14 (16384) | [8, 8, 8, 6, 5] | 5 | 15360 |
| 2^16 (65536) | [8, 8, 8, 5, 5, 5] | 6 | 64000 |

**Heuristic**: Keep L_i ≥ 5 for all dimensions. This ensures each dimension has enough resolution for the model to spread information effectively.

---

## 6. Comparison with VQ-VAE

| Aspect | VQ | FSQ |
|--------|----|----|
| Codebook | Learned K vectors in ℝ^D (D ≥ 512) | Implicit, fixed grid in ℝ^d (d < 10) |
| Quantization | Nearest neighbor in D-dim space | Independent rounding per scalar |
| Codebook params | K × D (e.g., 4096 × 512 = 2M) | 0 (implicit) |
| Auxiliary losses | Commitment + codebook/EMA | None |
| Codebook utilization | Degrades >2^11, requires tricks | ~100% at all sizes |
| Training complexity | EMA updates, reseeding, splitting | Just reconstruction loss |
| Representation dim | D ≥ 512 (high-dimensional) | d < 10 (low-dimensional) |

---

## 7. Why VQ Suffers Codebook Collapse

VQ defines a learnable Voronoi partition in high-dimensional space:
- Quantization `argmin` is non-differentiable → STE + auxiliary losses
- Commitment and codebook losses create conflicting optimization signals
- As K grows, many codewords receive no encoder assignments → "dead codes"
- Rich-get-richer: popular codes attract more updates, unpopular codes drift further away

**Mitigations** (all add complexity): EMA updates, codebook reseeding, code splitting, entropy penalties.

---

## 8. Why FSQ Avoids Collapse

- Fixed grid structure → every possible code has equal a priori "existence"
- Reconstruction loss gradients force the encoder to spread information across quantization bins
- No learned codebook vectors that can "die"
- Low-dimensional space (d < 10) ensures encoder can easily reach all regions

**Mathematical insight**: "VQ defines learnable Voronoi partition in high-dimensional space; FSQ relies on simple fixed grid partition in much lower-dimensional space."

---

## 9. MaskGIT Experiments (ImageNet 256×256)

Two-stage architecture:
- **Stage I**: Convolutional VQ-GAN/FSQ-GAN tokenizer
- **Stage II**: Masked transformer (BERT-style) for generation

### Results (codebook ~2^10)

| Method | Sampling FID↓ | Precision↑ | Recall↑ | Codebook Usage |
|--------|--------------|------------|---------|----------------|
| VQ (K=1024) | 4.509 | 0.860 | 0.465 | 81% |
| FSQ (L=[8,5,5,5]) | 4.534 | 0.864 | 0.453 | 100% |

Comparable generation quality with full codebook utilization.

### Codebook scaling behavior

| Codebook size | VQ Usage | FSQ Usage | VQ Recon FID | FSQ Recon FID |
|---------------|----------|-----------|--------------|---------------|
| 2^8 | ~95% | ~100% | Higher | Higher |
| 2^11 | ~85% | ~100% | Best VQ point | Improving |
| 2^14 | <50% | ~100% | Degrading | Still improving |
| 2^16 | <50% | ~100% | Poor | Best |

VQ peaks at ~2^11 then degrades. FSQ improves monotonically with codebook size.

---

## 10. UViM Experiments (Dense Prediction)

Transformer-based VQ-VAE for structured outputs (codebook size 4096, ~12 bits):

| Task | VQ | FSQ [7,5,5,5,5] | Δ |
|------|----|----|---|
| Panoptic Seg. (PQ↑) | 43.4 | 43.2 | -0.2 |
| Depth Est. (RMSE↓) | 0.468 | 0.473 | +0.005 |
| Colorization (FID↓) | 16.90 | 17.55 | +0.65 |

FSQ is slightly worse but eliminates all VQ complexity. Both achieve ~100% codebook usage.

---

## 11. Ablation Studies

### Dimensions vs. Levels
- Too few dimensions (d=1 or 2 with many levels): poor performance
- Too many dimensions (d=10+ with few levels): diminishing returns
- Sweet spot: d=3–6 with L_i ∈ [5, 8]

### Effect of Minimum Level
- L_i < 5: degraded performance (too coarse per dimension)
- L_i ≥ 5: consistent good performance
- Hypothesis: model needs ≥5 levels per dimension to encode meaningful variation

### Codebook Splitting (VQ trick applied to FSQ)
- VQ without splitting: 0.78% codebook usage, visual artifacts
- FSQ without splitting: 99% usage, no artifacts
- FSQ makes splitting unnecessary

---

## 12. Connection to Product Quantization

FSQ creates an **implicit product codebook**: the Cartesian product of per-dimension quantization ranges. Unlike explicit product quantization:
- No sub-codebook learning
- No distance computation
- Fixed structure determined by architecture choice (d, L)

---

## 13. Compression Cost Analysis

Proxy metric: compress discrete codes with autoregressive transformer using entropy coding.

- VQ: compression cost correlates with (low) codebook utilization — unused codes = wasted capacity
- FSQ: cost increases with codebook size, but generation quality (FID) saturates around 2^12

---

## 14. Implementation (Pseudocode)

```python
class FSQ:
    def __init__(self, levels: list[int]):
        # levels = [8, 5, 5, 5] for ~2^10 codebook
        self.levels = levels
        self.d = len(levels)
        # Encoder projects to d dimensions
    
    def bound(self, z, L_i):
        half_l = (L_i - 1) * (1 - 1e-3) / 2
        offset = 0.5 if L_i % 2 == 0 else 0.0
        shift = arctanh(offset / half_l)
        return tanh(z + shift) * half_l - offset
    
    def quantize(self, z):
        # z: [..., d] — last dim has d channels
        z_bounded = [self.bound(z[..., i], L) for i, L in enumerate(self.levels)]
        z_hat = [round_ste(zb) for zb in z_bounded]
        return stack(z_hat, dim=-1)
    
    def codes_to_indices(self, z_hat):
        # Mixed-radix encoding
        basis = cumprod([1] + self.levels[:-1])
        # Shift to non-negative: z_hat_i + floor(L_i/2)
        shifted = z_hat + floor(tensor(self.levels) / 2)
        return (shifted * basis).sum(dim=-1)
```

---

## 15. Key Properties for Implementation

| Property | Detail |
|----------|--------|
| Quantization type | Scalar (independent per dimension) |
| Codebook | Implicit (Cartesian product of levels) |
| Codebook parameters | 0 (no learned codebook) |
| Gradient method | Straight-through estimator on round() |
| Auxiliary losses | None needed |
| Representation dim | d < 10 (very low) |
| Levels per dim | Typically 5–8 |
| Codebook utilization | ~100% without tricks |
| Training stability | Excellent — no collapse, no special initialization |

---

## 16. Relevance to Weight Quantization

FSQ quantizes **activations/representations** in a generative model context. For weight quantization:
- Each weight (or group) is independently quantized to one of L levels → this IS essentially what EfficientQAT does
- EfficientQAT's uniform affine quantization: `ẑ = round(clamp((w - z)/s, 0, 2^n-1)) * s + z` maps each scalar to 2^n levels
- The key difference: FSQ uses fixed bounds (tanh), EfficientQAT learns scale s and zero_point z

**FSQ is closer to EfficientQAT than VQ is** — both are scalar quantization with STE. The distinction is:
- FSQ: fixed levels, applied to low-dim activation vectors in tokenizers
- EfficientQAT: learned affine parameters (s, z), applied to individual weight scalars in LLM layers
