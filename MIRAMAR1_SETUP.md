# EfficientQAT on miramar1

Machine specs: 24GB VRAM (A4500), 64GB RAM

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## wandb Setup

```bash
cp .env.example .env
# Edit .env and set your WANDB_API_KEY
```

## Download Llama-2-7B

```bash
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./models/Llama-2-7b-hf
```

Requires Hugging Face account with Llama 2 access approved.

## Training (unified CLI)

Run both phases with wandb logging:

```bash
python train.py \
  --model ./models/Llama-2-7b-hf \
  --net Llama-2 \
  --wbits 4 \
  --group_size 128 \
  --scheme uniform_affine \
  --dataset redpajama \
  --e2e_batch_size 2 \
  --gradient_accumulation_steps 16
```

Run only Block-AP:

```bash
python train.py \
  --model ./models/Llama-2-7b-hf \
  --net Llama-2 \
  --wbits 4 \
  --group_size 128 \
  --scheme uniform_affine \
  --phases block_ap
```

Run only E2E-QP (assumes Block-AP already completed):

```bash
python train.py \
  --model ./models/Llama-2-7b-hf \
  --net Llama-2 \
  --wbits 4 \
  --group_size 128 \
  --scheme uniform_affine \
  --phases e2e_qp \
  --dataset redpajama
```

## Manual phase commands (without unified CLI)

### Phase 1: Block-AP

```bash
CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
  --model ./models/Llama-2-7b-hf \
  --output_dir ./output/block_ap_log/Llama-2-7b-w4g128 \
  --net Llama-2 \
  --wbits 4 \
  --group_size 128 \
  --quant_lr 1e-4 \
  --weight_lr 1e-5 \
  --real_quant \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
  --save_quant_dir ./output/block_ap_models/Llama-2-7b-w4g128 \
  --wandb_project qat \
  --scheme uniform_affine
```

### Phase 2: E2E-QP

```bash
CUDA_VISIBLE_DEVICES=0 python main_e2e_qp.py \
  --quant_model_path ./output/block_ap_models/Llama-2-7b-w4g128 \
  --model_family Llama-2 \
  --wbits 4 \
  --group_size 128 \
  --learning_rate 1e-5 \
  --dataset redpajama \
  --dataset_format pt \
  --output_dir ./output/e2e_qp_models/Llama-2-7b-w4g128-redpajama \
  --do_train True \
  --pt_context_len 4096 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --logging_steps 1 \
  --save_strategy epoch \
  --training_strategy epochs \
  --evaluation_strategy steps \
  --eval_steps 64 \
  --max_train_samples 4096 \
  --num_train_epochs 1 \
  --eval_dataset_size 64 \
  --bf16 \
  --data_seed 42 \
  --max_grad_norm 0.3 \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
  --preprocessing_num_workers 16 \
  --do_ppl_eval \
  --wandb_project qat \
  --scheme uniform_affine
```

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
  --resume_quant ./output/e2e_qp_models/Llama-2-7b-w4g128-redpajama \
  --net Llama-2 \
  --wbits 4 \
  --group_size 128 \
  --output_dir ./output/inference_results/ \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

## Inference only (skip training)

```bash
huggingface-cli download ChenMnZ/Llama-2-7b-EfficientQAT-w4g128 --local-dir ./output/pre_quantized_models/Llama-2-7b-EfficientQAT-w4g128

CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
  --resume_quant ./output/pre_quantized_models/Llama-2-7b-EfficientQAT-w4g128 \
  --net Llama-2 \
  --wbits 4 \
  --group_size 128 \
  --output_dir ./output/inference_results/ \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

## Recommended configs for 24GB VRAM

The A450 with 24GB VRAM can handle larger batch sizes than the 16GB anjuna3 setup.

| Config | wbits | group_size | batch_size | grad_accum | context_len | Notes |
|--------|-------|------------|------------|------------|-------------|-------|
| w4g128 | 4     | 128        | 2          | 16         | 4096        | Default, fits comfortably |
| w3g128 | 3     | 128        | 2          | 16         | 4096        | Smaller model, more headroom |
| w2g128 | 2     | 128        | 4          | 8          | 4096        | Smallest model, can push batch |
| w2g64  | 2     | 64         | 2          | 16         | 4096        | More groups = more params |

## OOM troubleshooting

With 24GB you have more headroom than anjuna3 but can still OOM on w4g128 if batch is too large:
- Drop `--per_device_train_batch_size` to 1 and double `--gradient_accumulation_steps` to 32
- Reduce `--pt_context_len` to 2048 as a last resort
