# VQ-VAE: Neural Discrete Representation Learning

**Paper**: Neural Discrete Representation Learning  
**Authors**: Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu (DeepMind)  
**Year**: 2017 (NeurIPS)  
**arXiv**: [1711.00937](https://arxiv.org/abs/1711.00937)

---

## 1. Core Idea

VQ-VAE is a variational autoencoder that learns **discrete** latent representations via vector quantization. Unlike standard VAEs with continuous latents and fixed priors, VQ-VAE:
- Produces discrete codes (one-hot categorical)
- Learns the prior separately (autoregressive model)
- Avoids posterior collapse via the information bottleneck of discretization

---

## 2. Architecture

```
Input x → Encoder z_e(x) → Vector Quantization → z_q(x) → Decoder → Reconstruction x̂
                                    ↕
                          Embedding table e ∈ ℝ^{K×D}
```

- **Encoder**: Maps input x to continuous representation z_e(x) ∈ ℝ^D
- **Embedding space (Codebook)**: K embedding vectors e_1, ..., e_K ∈ ℝ^D
- **Decoder**: Reconstructs from quantized z_q(x)

---

## 3. Quantization Operation

The discrete latent is determined by nearest-neighbor lookup:

### Posterior (Equation 1):
```
q(z = k | x) = { 1  if k = argmin_j ||z_e(x) - e_j||_2
               { 0  otherwise
```

### Quantized representation (Equation 2):
```
z_q(x) = e_k,  where  k = argmin_j ||z_e(x) - e_j||_2
```

This is a deterministic, one-hot categorical posterior. The encoder output is replaced by the nearest codebook vector.

---

## 4. Loss Function (Equation 3)

```
L = -log p(x | z_q(x)) + ||sg[z_e(x)] - e||²₂ + β·||z_e(x) - sg[e]||²₂
```

Three terms:

| Term | Name | What it trains | Purpose |
|------|------|---------------|---------|
| `-log p(x \| z_q(x))` | Reconstruction loss | Encoder + Decoder (via STE) | Minimize reconstruction error |
| `\|\|sg[z_e(x)] - e\|\|²` | VQ loss (codebook loss) | Embedding vectors e | Move codebook entries toward encoder outputs |
| `β·\|\|z_e(x) - sg[e]\|\|²` | Commitment loss | Encoder | Prevent encoder output from growing unbounded |

Where:
- `sg[·]` = stop-gradient operator (identity forward, zero gradient backward)
- `β` = commitment cost coefficient (β = 0.25 in experiments; range 0.1–2.0 works)

---

## 5. Straight-Through Estimator (STE)

The quantization operation `argmin` is non-differentiable. VQ-VAE uses STE:

**Forward pass**: z_q(x) = nearest codebook entry  
**Backward pass**: Gradients are copied directly from decoder input to encoder output

```
∇_encoder L ≈ ∇_{z_q} L    (gradient copied unchanged)
```

This works because encoder and decoder share the same D-dimensional space, so the gradient direction is informative for the encoder.

---

## 6. Exponential Moving Average (EMA) Updates

Alternative to the VQ loss term (Term 2). Updates codebook entries as an online K-means:

```
N_i^(t)  = γ · N_i^(t-1) + (1 - γ) · n_i^(t)
m_i^(t)  = γ · m_i^(t-1) + (1 - γ) · Σ_j z_{i,j}^(t)
e_i^(t)  = m_i^(t) / N_i^(t)
```

Where:
- `N_i` = EMA of count of encoder outputs assigned to embedding i
- `m_i` = EMA of sum of encoder outputs assigned to embedding i
- `n_i^(t)` = number of encoder outputs in current batch assigned to i
- `γ` = decay rate (γ = 0.99)

With EMA, the loss becomes just: `L = -log p(x|z_q(x)) + β·||z_e(x) - sg[e]||²₂`

---

## 7. Prior Distribution

**During training**: Uniform prior p(z) = 1/K for all k  
- KL divergence = log K (constant, not optimized)

**For generation**: Learned autoregressive prior fit post-training:
- **Images**: PixelCNN over the discrete latent map
- **Audio**: WaveNet over the discrete sequence
- Joint training left as future work

---

## 8. Log-Likelihood Evaluation

```
log p(x) = log Σ_k p(x|z_k) · p(z_k)
         ≥ log p(x|z_q(x)) · p(z_q(x))   (lower bound)
```

At convergence, decoder assigns near-zero probability to z ≠ z_q(x), so the bound is tight.

---

## 9. Architecture Details

### Images (CIFAR-10 / ImageNet 128×128)

| Component | Details |
|-----------|---------|
| Encoder | 2 strided conv (stride 2, 4×4 kernel, ReLU) + 2 residual blocks (3×3, 256 hidden) |
| Decoder | 2 residual blocks (3×3) + 2 transposed convolutions |
| Latent | ImageNet: 128×128×3 → 32×32×1 (K=512) |
| Compression | ~42.6× spatial reduction |

### Audio (VCTK, 109 speakers)

| Component | Details |
|-----------|---------|
| Encoder | 6 strided convolutions (stride 2, window 4) |
| Downsampling | 64× temporal reduction |
| Latent | K=512, single feature map |
| Decoder | Conditioned on speaker one-hot embedding |
| Application | Speaker conversion without parallel data |

### Video (DeepMind Lab)

| Component | Details |
|-----------|---------|
| Input | 84×84×3 frames |
| Latent (stage 1) | 21×21×1 |
| Latent (stage 2) | 3 latent variables (27 bits total) |
| Application | Action-conditioned video prediction |

---

## 10. Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 2×10⁻⁴ |
| Batch size | 128 |
| Training steps | 250,000 (CIFAR-10) |
| β (commitment) | 0.25 |
| K (codebook size) | 512 |
| D (embedding dim) | matches encoder output channels |

---

## 11. Key Results

### Log-likelihood (bits/dim on CIFAR-10)

| Model | bits/dim |
|-------|----------|
| Continuous VAE | 4.51 |
| VQ-VAE (discrete) | 4.67 |
| VIMCO (discrete) | 5.14 |

First demonstration that discrete latent VAEs can match continuous VAEs.

### Speaker Conversion
- VQ-VAE latents capture content (phonemes) while discarding speaker identity
- Reconditioning on different speaker embedding achieves conversion without parallel data

### Unsupervised Phoneme Discovery
- 49.3% accuracy mapping 128 discrete codes to 41 phonemes
- Random baseline: 7.2%

---

## 12. Posterior Collapse & Information Bottleneck

**Problem with standard VAEs**: When decoder is powerful (autoregressive), it learns to ignore z entirely, collapsing posterior to prior.

**VQ-VAE solution**: 
- Discrete bottleneck forces information through z
- No KL term to minimize (it's constant = log K)
- Two-stage model: VQ-VAE learns representation, then PixelCNN models the prior
- Demonstrated with PixelCNN decoder: impossible for standard VAE but VQ-VAE maintains meaningful latents

---

## 13. Key Properties for Implementation

| Property | Detail |
|----------|--------|
| Quantization type | Vector (D-dimensional codebook lookup) |
| Gradient method | Straight-through estimator |
| Codebook update | EMA (preferred) or VQ loss gradient |
| Auxiliary losses | Commitment loss (β=0.25) |
| Codebook size | K entries, each D-dimensional |
| Output space | One-hot over K categories per spatial position |
| Training stability | Sensitive to β; EMA more stable than gradient-based codebook updates |
| Known issue | Codebook collapse (dead codes) when K is large |

---

## 14. Relevance to Weight Quantization

VQ-VAE quantizes **activations/representations** not weights. However, the core VQ mechanism could be adapted for weight quantization:
- Group weights into vectors → find nearest codebook entry
- Each group stores only the codebook index (log₂K bits)
- Codebook entries are shared across all groups
- STE enables gradient-based training of the encoder (here: weight → index mapping)

**Key difference from EfficientQAT's approach**: VQ uses a learned codebook in D-dimensional space (vector), while EfficientQAT uses uniform affine scalar quantization (per-scalar, fixed grid defined by scale+zero_point).
