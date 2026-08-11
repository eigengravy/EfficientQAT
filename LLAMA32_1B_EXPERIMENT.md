# Llama 3.2 1B: EfficientQAT versus bounded DDCL

This is the primary experiment. It compares only standard EfficientQAT and
bounded DDCL during Block-AP. Both methods then use the identical EfficientQAT
E2E-QP phase and the identical fixed-width packed representation.

Run this on an NVIDIA CUDA machine, not macOS.

## Model

Download the base checkpoint:

```bash
huggingface-cli download meta-llama/Llama-3.2-1B \
  --local-dir ./models/Llama-3.2-1B
```

## Smoke test: W4G128

Standard EfficientQAT:

```bash
python train.py \
  --model ./models/Llama-3.2-1B \
  --net Llama-3.2-1B \
  --scheme uniform_affine \
  --wbits 4 \
  --group_size 128 \
  --seed 1 \
  --train_size 128 \
  --val_size 16 \
  --e2e_train_size 128 \
  --e2e_val_size 16
```

Bounded DDCL:

```bash
python train.py \
  --model ./models/Llama-3.2-1B \
  --net Llama-3.2-1B \
  --scheme ddcl \
  --ddcl_lambda 1e-5 \
  --wbits 4 \
  --group_size 128 \
  --seed 1 \
  --train_size 128 \
  --val_size 16 \
  --e2e_train_size 128 \
  --e2e_val_size 16
```

## Main comparison

After both smoke tests complete, use W3G128 with the full 4,096/64 calibration
split and seeds 1, 2, and 3. Keep all non-scheme arguments identical.

DDCL ablations use `--ddcl_lambda 0`, `1e-5`, and `1e-4`. W2G64 is a later
stress test, not the first validation target.

Artifacts are isolated automatically under:

```text
output/Llama-3.2-1B/<scheme>/[lambda-*]/w<bits>g<group>/seed-<seed>/
```

For an E2E-QP-only continuation, provide the Block-AP model explicitly:

```bash
python train.py \
  --model ./models/Llama-3.2-1B \
  --net Llama-3.2-1B \
  --scheme ddcl \
  --ddcl_lambda 1e-5 \
  --wbits 3 \
  --group_size 128 \
  --seed 1 \
  --phases e2e_qp \
  --block_ap_model_path /absolute/path/to/block_ap_model
```
