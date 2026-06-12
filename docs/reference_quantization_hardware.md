# Quantization in LLMs: Techniques & Hardware Co-Design

**Source**: [Eric X. Liu — Quantization in LLMs](https://ericxliu.me/posts/quantization-in-llms/)  
**Focus**: Comprehensive survey of quantization methods with emphasis on hardware support requirements for real-world speedups.

---

## 1. The Central Insight: Hardware Support Determines Real Speedup

Quantization reduces model size, but **actual inference speedup requires hardware that natively executes the quantized format**. Without hardware support:
- K-means quantization needs lookup tables → no standard accelerator support
- Binary/ternary methods enable bitwise ops → lack widespread hardware acceleration
- Integer quantization (INT8/INT4) benefits from decades of processor optimization

**The implication**: When choosing a quantization scheme, the question isn't just "what accuracy can I get at N bits?" but "does my target hardware have native arithmetic for this format?"

Modern formats like MXFP4 represent **co-designing models and silicon for maximum performance** — quantization schemes are only valuable when hardware can execute them efficiently.

---

## 2. Quantization Schemes

### 2.1 K-Means (Non-Uniform) Quantization

Groups weights into K clusters, stores centroid codebook + per-weight index.

- **Pro**: Adapts to weight distribution, good compression ratio
- **Con**: Inference requires codebook lookup per weight → no standard hardware acceleration
- **Practical use**: Storage/transmission compression, not runtime speedup

### 2.2 Linear/Affine Quantization (Dominant Method)

```
r = S · (q - Z)
```

- `r` = reconstructed real value
- `q` = quantized integer
- `S` = scale factor (step size)
- `Z` = zero-point offset

**Why dominant**: Maps directly to integer multiply-accumulate (MAC) operations that GPUs/TPUs execute natively. Decades of hardware optimization for integer arithmetic.

### 2.3 Binary and Ternary Quantization

- **Binary** (±1): BinaryConnect, XNOR-Net — enables bitwise XNOR + popcount
- **Ternary** (±1, 0): TWN, TTQ — adds zero for sparsity, learnable scaling

**Status**: Extreme compression but substantial accuracy loss on complex models. Limited practical deployment due to accuracy gap AND lack of widespread hardware acceleration.

---

## 3. Quantization Granularity

| Level | Scale/ZP per... | Overhead | Precision | Use Case |
|-------|----------------|----------|-----------|----------|
| Per-tensor | Entire weight matrix | Minimal | Coarse | Simple, fast |
| Per-channel | Output channel | Low | Good | Standard for CNNs |
| Per-group | Group of G weights | Moderate | Best | Essential for LLMs |

**Key insight for LLMs**: Per-channel and group quantization are **indispensable** due to heterogeneous weight distributions. A single scale factor per tensor cannot handle the outlier-heavy distributions in transformer weights.

---

## 4. Calibration Methods (PTQ)

| Method | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| Min-Max | Use observed min/max as clipping bounds | Simple | Sensitive to outliers |
| EMA | Smoothed running statistics over calibration data | More stable | Requires multiple passes |
| KL-Divergence | Minimize information loss between original & quantized | Principled | Computationally expensive |
| MSE | Minimize reconstruction error directly | Direct optimization | May not correlate with task loss |
| AdaRound | Optimize individual rounding decisions | Fine-grained | Slow |

---

## 5. QAT and Straight-Through Estimator

QAT integrates quantization into training. The model learns weights robust to low-precision:

```
Forward:  w_q = quantize(w)    → use quantized weights
Backward: ∂L/∂w ≈ ∂L/∂w_q    → STE copies gradient through non-differentiable round()
```

STE is critical: it approximates the gradient of the non-differentiable quantization function as identity within the valid range, zero outside (clipped).

**QAT vs PTQ**: QAT achieves higher accuracy especially for aggressive low-bit (≤4-bit) quantization, but requires training compute.

---

## 6. Modern LLM Quantization Methods

### 6.1 GPTQ (Post-Training, Layer-wise)

- Quantizes weights layer-by-layer minimizing output MSE
- **Key insight**: Uses Hessian (second-order) information to identify critical weights
- Processes weights in order of sensitivity (inverse Hessian diagonal)
- Near-QAT quality with only small calibration set, no retraining
- Enables 4-bit LLM inference on consumer hardware

### 6.2 AWQ (Activation-Aware Weight Quantization)

- Identifies "important" weights by examining activation magnitudes
- **Key insight**: Not all weights equally important — those interacting with large activations matter more
- Scales important weights up before quantization to protect precision
- No full retraining needed

### 6.3 SpQR (Sparsity + Quantization)

- Hybrid: prune redundant weights first, then quantize remaining
- Special handling for outlier weights (stored at higher precision)
- Greater compression than either technique alone

### 6.4 QLoRA (Quantized Low-Rank Adaptation)

- Freeze base model at 4-bit quantization
- Fine-tune only small low-rank adapters (LoRA) in full precision
- Enables fine-tuning 65B models on single 48GB GPU
- Key enabler: NormalFloat4 (NF4) data type optimized for normally-distributed weights

---

## 7. Multi-Level Scaling: The Hardware Format Landscape

The key formula for hierarchical quantization:
```
r = (q - z) · s_l0 · s_l1
```

Multiple scale factors at different granularities multiply together.

### Comparison Table

| Scheme | Base Format | Scale L0 | Scale L1 | Effective Bits | Notes |
|--------|------------|----------|----------|----------------|-------|
| Per-Channel | INT4 | FP16 (per-channel) | — | 4.0 | Baseline |
| VSQ | INT4 | UINT4 (per-16) | FP16 (per-channel) | 4.25 | Two-level hierarchy |
| MX4 | S1M2 (3-bit) | E1M0 (per-2) | E8M0 (per-32) | 4.0 | Floating-point based |
| MX6 | S1M4 (5-bit) | — | E8M0 (per-32) | 6.0 | Better precision |
| MX9 | S1M7 (8-bit) | — | E8M0 (per-32) | 9.0 | Near-FP precision |

### Why Multi-Level Matters

- Single-level INT4 → 16 possible values, fixed dynamic range
- Multi-level (MX formats) → wider dynamic range via shared exponents
- Critical for LLMs: outlier weights/activations break fixed-point schemes
- MX formats handle outliers through floating-point representation with shared exponents

---

## 8. Hardware-Optimized Formats

### NVIDIA FP8 (H100/Blackwell)

Two variants:
- **E4M3**: 4 exponent bits, 3 mantissa → wider range, less precision
- **E5M2**: 5 exponent bits, 2 mantissa → even wider range, coarser precision

Native tensor core support on H100+. Direct matrix multiplication without dequantization overhead.

### MXFP4 (Microscaling FP4)

- Block-wise floating point: groups of elements share an exponent
- 4-bit per element with shared scaling → effective ~4-5 bits
- Designed for next-gen silicon (NVIDIA Blackwell, custom ASICs)
- Key benefit: FP-like dynamic range at INT4-like density

### GGUF K-Quants (llama.cpp)

- Hierarchical integer quantization for CPU inference
- Multiple "K-quant" types (Q4_K_M, Q5_K_S, Q6_K, etc.)
- Each block of weights has its own scale/min at different granularities
- Optimized for ARM NEON / x86 AVX2 integer SIMD

---

## 9. The Hardware Support Spectrum

| Format | Hardware Support | Speedup Reality |
|--------|-----------------|-----------------|
| FP16 | Universal (all modern GPUs) | Baseline |
| INT8 | NVIDIA Tensor Cores, Intel VNNI, ARM | Real 2× speedup |
| INT4 | NVIDIA (via packed ops), specialized | Real 2-4× memory, variable compute speedup |
| FP8 | H100+ Tensor Cores | Real 2× over FP16 |
| MXFP4 | Blackwell+ | Expected 2-4× |
| Binary/Ternary | No mainstream accelerator | Theoretical only |
| K-means/VQ | No mainstream accelerator | Memory savings only |

**Critical distinction**: 
- **Memory savings** = always proportional to bit reduction (loading less data from DRAM)
- **Compute speedup** = only when hardware has native low-precision arithmetic units

For memory-bandwidth-bound operations (which LLM inference typically is), even without compute speedup, reduced memory traffic = real latency reduction. But the best gains come when BOTH memory AND compute benefit.

---

## 10. Key Takeaways for Implementation

1. **Linear affine quantization remains foundational** — all modern methods build on `r = S·(q - Z)`

2. **Group quantization is essential for LLMs** — heterogeneous weight distributions demand per-group (or finer) scaling

3. **PTQ methods (GPTQ, AWQ) are practical** — near-QAT quality without retraining, enabling consumer hardware deployment

4. **The future is co-design** — new hardware (H100 FP8, Blackwell MXFP4) and new formats are designed together for maximum throughput

5. **Format choice = deployment target** — pick the quantization format that your inference hardware natively supports:
   - A100: INT8 tensor cores, INT4 via packing
   - H100: FP8 tensor cores, INT8
   - CPU (llama.cpp): GGUF K-quants with SIMD
   - Edge: INT4/INT8 via specialized NPUs

6. **Mixed-precision is the pragmatic path** — different layers can use different bit-widths based on sensitivity analysis

---

## 11. Relevance to This Project (EfficientQAT)

EfficientQAT uses **uniform affine quantization with learned scale and zero_point** — this is squarely in the "linear/affine quantization" family that has the best hardware support story.

### Current approach alignment with hardware:
- INT4 group quantization → matches NVIDIA INT4 packed tensor core ops
- Per-group scale factors → compatible with GPTQ/AWQ deployment toolchains
- Triton dequant kernels → custom GPU kernels for memory-bandwidth-bound inference

### If implementing alternative schemes:
- **VQ/K-means**: Would lose hardware acceleration. Memory savings only, no compute speedup. Dequantization becomes a gather operation (codebook lookup) rather than a simple multiply-add.
- **FSQ**: Same scalar quantization as current approach — equally hardware-friendly. The fixed levels vs learned scale is a training-time distinction; at inference, both produce integer weights with scale factors.
- **DDCL-style variable precision**: Would require custom kernels for mixed-width packing. No current hardware native support for variable-bit arithmetic within a tensor.
- **Binary/Ternary**: Extreme compression but no mainstream hardware path. XNOR + popcount exists on CPUs but not GPU tensor cores.

### The practical rule:
> If you want actual inference speedup (not just memory savings), your quantized format must map to operations the target hardware executes natively. INT4/INT8/FP8 with group scaling is the sweet spot for current GPU generations.
