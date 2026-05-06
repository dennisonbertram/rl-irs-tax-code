# RunPod Cloud GPU Training Research

**Date:** 2026-03-29
**Context:** Evaluating cloud GPU training for IRS tax code RL pipeline. Currently training Qwen 2.5 3B locally on Apple M4 Max 128GB with MLX. Considering scaling to Qwen 3.5 27B. Pipeline: SFT -> DPO -> GRPO.

---

## 1. GPU Options and Pricing

RunPod offers 30+ GPU models across two tiers: **Community Cloud** (cheaper, third-party hosted) and **Secure Cloud** (SOC2-compliant data centers, ~$0.10-$0.40/hr premium).

### Key GPUs for Our Use Case

| GPU | VRAM | Community On-Demand | Spot/Low | Secure Cloud |
|-----|------|-------------------|----------|--------------|
| RTX 3090 | 24GB | $0.22/hr | $0.11/hr | N/A |
| RTX 4090 | 24GB | $0.34/hr | $0.20/hr | $0.61/hr |
| RTX A6000 | 48GB | $0.33/hr | $0.25/hr | ~$0.50/hr |
| L40S | 48GB | $0.79/hr | $0.40/hr | ~$1.00/hr |
| A100 SXM | 80GB | $1.39/hr | $0.79/hr | $1.49/hr |
| A100 PCIe | 80GB | $1.19/hr | $0.60/hr | $1.39/hr |
| H100 PCIe | 80GB | $1.99/hr | $1.50/hr | $2.39/hr |
| H100 SXM | 80GB | $2.69/hr | $1.50/hr | $3.09/hr |
| H200 | 141GB | $3.59/hr | N/A | ~$4.00/hr |

### Storage Pricing
- **Container Disk:** $0.10/GB/month (ephemeral, lost when pod stops)
- **Volume Disk (running):** $0.10/GB/month
- **Volume Disk (idle):** $0.20/GB/month
- **Network Volume (<1TB):** $0.07/GB/month
- **Network Volume (>1TB):** $0.05/GB/month
- **No data ingress/egress fees** -- this is a significant advantage

---

## 2. GPU Requirements for Our Use Case

### Qwen 2.5 3B with LoRA (bf16)
- **VRAM needed:** ~10-14GB for LoRA fine-tuning
- **Minimum GPU:** RTX 3090 (24GB) -- plenty of headroom
- **Sweet spot:** RTX 4090 (24GB) -- fastest consumer GPU, $0.34/hr
- **Budget option:** RTX 3090 at $0.22/hr or spot at $0.11/hr

### Qwen 3.5 27B with LoRA (bf16)
- **VRAM needed:** ~56GB for bf16 LoRA (model weights ~54GB in bf16)
- **Minimum GPU:** A100 80GB SXM ($1.39/hr) -- recommended
- **Alternatives:** H100 80GB ($1.99-2.69/hr) for faster training
- **QLoRA (4-bit):** Could fit on 24GB GPU but **not recommended for Qwen models** -- quantization degrades quality more than usual
- **Note:** 48GB GPUs (L40S, A6000) are borderline and may OOM with larger batch sizes

### Training Speed Comparison (estimated)

For the 3B model (SFT, 1500 iters):
- M4 Max (MLX): ~40 minutes (your current baseline)
- RTX 4090 (PyTorch/CUDA): ~8-12 minutes (3-5x faster)
- A100 80GB: ~6-10 minutes (4-6x faster)
- H100 SXM: ~4-7 minutes (6-10x faster)

The speedup comes from CUDA's mature optimization + higher memory bandwidth on datacenter GPUs.

---

## 3. How RunPod Works

### Pods (Primary for Training)
- Spin up a GPU instance from a template or custom Docker image
- Access via **SSH**, **JupyterLab**, **VS Code/Cursor remote**, or web terminal
- Per-second billing -- only pay for actual compute time
- Can stop a pod (keep volume, stop billing compute) and restart later

### Storage Architecture
1. **Container Disk:** Ephemeral, lost on stop. Good for temp files.
2. **Volume Disk:** Persists with the pod lease. Mounted at `/workspace`. Keeps your scripts, checkpoints, etc.
3. **Network Volume:** Independent persistent storage. Can attach to multiple pods. Best for datasets that outlive any single pod.

### Workflow for Training
1. Create a Network Volume in your preferred data center region
2. Upload training data to the Network Volume (via S3 API or runpodctl)
3. Launch a pod with a PyTorch template, attach the Network Volume
4. SSH in, install any extra deps, run training
5. Checkpoints save to `/workspace` or Network Volume
6. Stop pod when done -- volume persists, compute billing stops

### Serverless (Not Relevant for Training)
Serverless endpoints are for inference, not training. Stick with Pods.

---

## 4. PyTorch / HuggingFace Support

### Official PyTorch Template
RunPod provides an **official PyTorch template** that comes pre-configured with:
- CUDA toolkit
- PyTorch (latest stable)
- JupyterLab
- Common ML libraries

### What You'd Need to Install (pip install on top)
```bash
pip install transformers datasets trl peft accelerate bitsandbytes wandb
```
This takes ~2-3 minutes on first pod launch. You can bake it into a custom template to avoid reinstalling each time.

### Custom Template Option
Create a custom Docker image with everything pre-installed:
- Base: `runpod/pytorch:2.x-cuda12.x`
- Add: transformers, trl, peft, accelerate, your training scripts
- Push to Docker Hub, use as template
- Every future pod starts instantly with your full environment

### MLX to PyTorch Migration
Switching from MLX to PyTorch for training is straightforward:
- HuggingFace `trl` library supports SFT, DPO, and GRPO natively with PyTorch
- LoRA via `peft` library is drop-in
- Your dataset format (JSONL) works identically
- The main work is rewriting training scripts from MLX's API to HuggingFace's `trl.SFTTrainer`, `trl.DPOTrainer`, `trl.GRPOTrainer`

---

## 5. Data Transfer

### For Your 357K JSONL Dataset (~2GB estimated)

**Recommended: Upload to HuggingFace Hub or S3, then pull into pod**
```bash
# From inside the pod:
huggingface-cli download your-username/irs-tax-dataset --local-dir /workspace/data
# or
wget https://your-s3-bucket.s3.amazonaws.com/train.jsonl -O /workspace/data/train.jsonl
```

**Alternative Methods:**
1. **runpodctl** -- RunPod's CLI tool for direct local-to-pod transfer. No port config needed.
2. **SCP/rsync** -- Standard SSH-based transfer: `scp -P <port> train.jsonl root@<pod-ip>:/workspace/data/`
3. **Network Volume S3 API** -- Upload directly to a Network Volume via S3-compatible API without needing a running pod
4. **git clone** -- If your data + scripts are in a repo (with Git LFS for large files)
5. **Drag and drop** -- Via JupyterLab or VS Code web interface (slow for large files)

**Best Practice:** Host data on a Network Volume. Upload once, attach to any pod. No re-uploading when you start/stop pods.

---

## 6. Provider Comparison

| Feature | RunPod | Lambda Labs | Vast.ai | Modal | Together.ai |
|---------|--------|-------------|---------|-------|-------------|
| **H100 price** | $1.99-2.69/hr | $2.89/hr | $1.87/hr | ~$3.50/hr | N/A (API only) |
| **A100 price** | $1.19-1.39/hr | $1.29/hr | $0.90-1.20/hr | ~$2.00/hr | N/A |
| **RTX 4090 price** | $0.34/hr | N/A | $0.24-0.60/hr | N/A | N/A |
| **Spot/preemptible** | Yes | No | Yes (marketplace) | No | N/A |
| **Ease of use** | High | High | Medium | High (code-first) | Very High |
| **Reliability** | Good | Excellent | Variable | Excellent | Excellent |
| **Persistent storage** | Yes (Network Vol) | Yes | Limited | No (ephemeral) | N/A |
| **SSH access** | Yes | Yes | Yes | No (code deploy) | No |
| **Templates** | Yes | Yes | Community | Docker/code | N/A |
| **Billing** | Per-second | Per-hour | Per-hour | Per-second | Per-token |
| **Ingress/Egress fees** | None | None | None | None | N/A |

### Verdict by Use Case

- **Best overall for our use case: RunPod** -- Good balance of price, flexibility, and ease of use. Network Volumes let you persist data cheaply between training runs. Per-second billing is ideal for iterative training.
- **Cheapest raw price: Vast.ai** -- P2P marketplace, lowest prices, but inconsistent reliability. GPU could disappear mid-training. Good for non-critical experimentation.
- **Most reliable: Lambda Labs** -- Premium experience, but no spot pricing and limited GPU availability. Often has waitlists for A100/H100.
- **Best for pure code workflows: Modal** -- Python-native, no SSH. Great for inference/serving, less ideal for interactive training iteration.
- **Together.ai** -- API-based fine-tuning service. Simplest but least flexible. Good if you want zero infrastructure management.

---

## 7. Estimated Costs for Our Pipeline

### 3B Model (Qwen 2.5 3B) on RTX 4090 ($0.34/hr community)

| Stage | Iters | Est. Time (CUDA) | Est. Cost |
|-------|-------|-------------------|-----------|
| SFT | 1500 | ~10-15 min | $0.06-0.09 |
| DPO | 1500 | ~15-25 min | $0.09-0.14 |
| GRPO | 1000 | ~20-30 min | $0.11-0.17 |
| **Total** | | **~45-70 min** | **$0.26-0.40** |

On spot RTX 4090 ($0.20/hr): **~$0.15-0.23** per full pipeline run.

### 27B Model (Qwen 3.5 27B) on A100 80GB SXM ($1.39/hr community)

| Stage | Iters | Est. Time (CUDA) | Est. Cost |
|-------|-------|-------------------|-----------|
| SFT | 1500 | ~2-4 hours | $2.78-5.56 |
| DPO | 1500 | ~3-5 hours | $4.17-6.95 |
| GRPO | 1000 | ~3-6 hours | $4.17-8.34 |
| **Total** | | **~8-15 hours** | **$11.12-20.85** |

On spot A100 ($0.79/hr): **~$6.32-11.85** per full pipeline run.

On H100 SXM ($2.69/hr, ~1.5-2x faster): **~$10.76-20.15** (faster but similar cost due to higher rate).

### Storage Costs (ongoing)
- 50GB Network Volume for data + checkpoints: ~$3.50/month
- Idle volume between runs: negligible at these sizes

### Monthly Budget Estimates

| Scenario | GPU | Runs/Month | Est. Monthly Cost |
|----------|-----|-----------|-------------------|
| 3B iteration (heavy) | RTX 4090 spot | 50 runs | ~$10-12 |
| 3B iteration (light) | RTX 4090 spot | 10 runs | ~$2-3 |
| 27B iteration (heavy) | A100 spot | 10 runs | ~$63-119 |
| 27B iteration (light) | A100 spot | 3 runs | ~$19-36 |
| Mixed (3B dev + 27B final) | Mixed | Typical | ~$30-80 |

---

## 8. Recommendations

### For Immediate Use (3B Model)
1. **Start with RunPod Community Cloud, RTX 4090** at $0.34/hr (or spot at $0.20/hr)
2. Use the official PyTorch template
3. Create a Network Volume, upload your JSONL data once
4. Port your MLX training scripts to PyTorch/trl -- this is the main upfront work
5. Expected: full SFT+DPO+GRPO pipeline in under 1 hour, costing ~$0.25-0.40

### For 27B Model
1. **A100 80GB SXM** on spot ($0.79/hr) is the best value
2. Full pipeline will cost ~$6-12 per run on spot
3. Consider Unsloth for 1.5x training speedup and 50% less VRAM
4. Do NOT use QLoRA for Qwen -- stick with bf16 LoRA

### Migration Steps
1. Rewrite training scripts: MLX -> PyTorch/trl (1-2 days of work)
2. Create RunPod account, add $25 credit to start
3. Create Network Volume, upload data
4. Test 3B pipeline first to validate
5. Scale to 27B once scripts are proven

### Key Advantages Over Local M4 Max
- **3-10x faster training** depending on GPU choice
- **27B model becomes feasible** (needs 56GB VRAM, M4 Max has unified 128GB but MLX is much slower for this size)
- **Pay only for compute time** -- no idle hardware cost
- **Scale up/down** -- use cheap GPUs for iteration, expensive ones for final runs
- **Cost is trivial** -- full 3B pipeline costs less than a coffee

---

## Sources
- [RunPod Pricing](https://www.runpod.io/pricing)
- [RunPod GPU Pricing via ComputePrices.com](https://computeprices.com/providers/runpod)
- [RunPod GPU Pricing Breakdown (Northflank)](https://northflank.com/blog/runpod-gpu-pricing)
- [RunPod Pods Documentation](https://docs.runpod.io/pods/overview)
- [RunPod Data Transfer Guide](https://www.runpod.io/blog/transfer-data-into-runpod)
- [RunPod GPU Types Reference](https://docs.runpod.io/references/gpu-types)
- [GPU Cloud Comparison 2026 (Northflank)](https://northflank.com/blog/cheapest-cloud-gpu-providers)
- [Qwen Fine-tuning with Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [RunPod LLM Fine-Tuning GPU Guide](https://www.runpod.io/blog/llm-fine-tuning-gpu-guide)
