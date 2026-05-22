# Module 01 — Tools and Environment Setup

> Part of [Part 0 — Mental Model](../). Reading time: ~30 minutes. Compute cost: ~$0–1 (one short cloud-pod session to verify the setup works).

## What you'll have at the end

- A local Python environment that can run every CPU-only experiment in this course.
- A cloud GPU you can spin up in under 2 minutes when you need it.
- Weights & Biases and HuggingFace working from your first training run.
- A working `torchrun` sanity check that proves your distributed setup is sane.
- A clear picture of what each remaining module will cost you in dollars.

## The workflow pattern: develop offline, run online

This is the most important habit in this course. Internalize it before anything else.

```
LOCAL (your laptop)                           CLOUD (RunPod / Lambda)
─────────────────                             ──────────────────────
- write code                                  - rent GPU
- read code                                   - sync code (git pull)
- inspect tensors                  → → → →    - launch with torchrun
- debug shapes                                - watch W&B from laptop
- run on tiny tensors                         - download checkpoints
- run unit tests                              - shut down pod
- think                                       (← back to local)
- $0                                          ($)
```

The cloud GPU is *expensive and impatient*. Every minute it's running while you type in your IDE, you're burning dollars. The local environment is *cheap and patient* — that's where you write, read, think, and break things.

**A bad workflow:** spin up a $2/hr GPU, code on it directly via VSCode SSH, get distracted, leave it running, run something, it fails, you debug, the GPU keeps running. $50 evaporates in 24 hours.

**A good workflow:** write everything locally, get the code 95% right with shape checks and tiny-tensor smoke tests, push to git, spin up the GPU, pull, launch, monitor on W&B from your laptop, shut down the moment training finishes.

Every module in this course is designed to fit this pattern.

## Local environment setup

**Python:** 3.11 (3.10–3.12 all fine).

**Package manager:** we recommend [`uv`](https://github.com/astral-sh/uv) — fast, modern, no surprises. Plain `venv + pip` works fine if you prefer.

### Install with `uv` (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Install with `venv + pip` (fallback)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify locally (no GPU needed)

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You should see something like `2.x.x False` on a CPU-only laptop. The `False` is fine — locally you don't need CUDA. The cloud GPU will have it.

## Cloud GPU setup

We recommend **RunPod** for this course. Simplest mental model: pick a GPU, pick an image, get a Jupyter URL and SSH, shut it down when done. Pay by the second.

### RunPod (recommended)

1. Sign up at [runpod.io](https://runpod.io). Add a credit card. Top up with $25 to start.
2. Templates → **"RunPod PyTorch 2.x"** (latest version, CUDA 12).
3. GPU → **A100 80GB** for serious runs; **A40 48GB** for lighter modules; **A100 40GB** if 80GB is unavailable.
4. Storage → 50 GB persistent volume. Cheap, survives pod restarts, worth it.
5. Deploy. SSH in. Verify:

```bash
nvidia-smi          # should show the A100
nvcc --version      # CUDA 12.x
```

### Lambda Labs (alternative)

Cleaner UI, slightly more expensive, harder to get A100s on demand. Same workflow as RunPod.

### What NOT to use

- **AWS / GCP / Azure** — billing complexity, idle-instance traps, IAM rabbit holes. Don't.
- **Colab free tier** — random disconnects, no persistent storage, T4 GPUs. Useless for this course.
- **Colab Pro+** — usable but you don't control the GPU, sessions die, can't run multi-day jobs. Only if you literally cannot use RunPod.
- **Consumer GPUs (3090/4090)** — work for some modules, but the course assumes A100-class HBM. You'll hit memory walls.

### Cost discipline

| Habit | Why |
|---|---|
| Stop the pod when not training. | A pod sitting idle costs the same as a pod training. RunPod bills by the second. |
| Use community/spot pods for everything except final checkpoints. | ~50% cheaper. They can be preempted, but our scripts checkpoint frequently. |
| Sync code via git, not SCP. | Forces you to commit. Cheap accountability for what's running on the pod. |
| Never do CPU-bound work on the GPU pod. | Tokenization, data filtering, plotting — do them locally. You're paying A100 rates for an idle GPU otherwise. |

## Accounts you need

Three free accounts. Set them up now.

| Service | Why | Setup |
|---|---|---|
| **HuggingFace** ([hf.co](https://huggingface.co)) | Source of datasets (FineWeb-Edu) and pretrained models (Qwen 3.6 1.7B). | Account → Settings → Access Tokens → create a read token. Save it. |
| **Weights & Biases** ([wandb.ai](https://wandb.ai)) | Experiment tracking. Loss curves, gradient norms, throughput. Free for individuals. | Account → Settings → API key. Save it. |
| **GitHub** | Sync your code to the cloud pod. | Already have one. |

On the cloud pod, log in once per pod:

```bash
huggingface-cli login   # paste HF token
wandb login             # paste W&B key
```

## The `torchrun` sanity check

Every training script in this course launches with `torchrun`. Even on a single GPU. The reasons are in [Module 00 (rule 4)](../00-what-we-are-building/) and the deep dive lives in Module 10. The pattern starts here.

Run this on your cloud GPU (from the repo root):

```bash
cd part-0-mental-model/01-tools-and-environment
torchrun --nproc_per_node=1 hello.py
```

You should see something like:

```
[rank 0 / world_size 1] hello from cuda:0
[rank 0] all-reduce check: tensor sum = 1.0 (expected 1.0) ✓
```

If your pod has more than one GPU, verify multi-GPU works too:

```bash
torchrun --nproc_per_node=2 hello.py
```

> If you see `can't open file '…/hello.py': [Errno 2] No such file or directory`, you ran torchrun from the repo root instead of from this module's folder. `cd` first; every per-module script in this course is launched from inside its own module directory.

Two ranks, all-reduce sums to 2.0 (each rank contributes 1.0). If it doesn't, your NCCL setup is broken — fix it now, not in Part 3 with real training running.

The full script is in [`hello.py`](hello.py). Read it. It's ~30 lines and contains the boilerplate every training script in this course will use.

## Compute budget, per module

Honest cost estimates assuming A100 80GB community pods on RunPod (~$1.50/hr). Running every module yourself lands in the **$35–60** range; using pre-run checkpoints for the two expensive ones (Modules 11 and 15) drops it to **~$25**.

| Module | What runs | Estimate |
|---|---|---|
| 00, 01, 02, 03 | Concepts + tokenization (CPU) | ~$0 |
| 04, 05, 06, 07 | Architecture forward passes (CPU or short GPU) | ~$0–2 |
| 08, 09, 10 | Training-loop walkthroughs on tiny model | ~$1–3 |
| **11** | Pretraining demo on FineWeb-Edu | **~$10–15** |
| 12 | Post-training landscape (concepts) | $0 |
| 13 | SFT on Qwen 3.6 1.7B | ~$3–5 |
| 14 | DPO on Qwen 3.6 1.7B | ~$3–5 |
| **15** | GRPO on GSM8K subset | **~$10–15** |
| 16 | Distillation (offline + on-policy + SDFT) | ~$5–10 |
| 17 | Evaluation runs | ~$2–4 |
| 18, 19 | Reading + reflection | $0 |
| **Total** | | **~$35–60** |

## Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available() == False` on the pod | CPU-only torch wheel installed | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `NCCL error` on torchrun | Wrong CUDA version, or `--nproc_per_node` exceeds GPU count | Match `--nproc_per_node` to `nvidia-smi` |
| `OOM` immediately | Batch too large or model too large for GPU | Reduce batch, enable gradient checkpointing, or pick a bigger GPU |
| `wandb` runs silently / no logs | Not logged in, or `WANDB_MODE=offline` set | `wandb login`; verify `WANDB_MODE=online` |
| Training feels slow but GPU is at 30% | Data loading bottleneck, not GPU bottleneck | Module 11 has the playbook — `nvidia-smi dmon` is your friend |

## Next

[Module 02 — The Corpus](../../part-1-data/02-the-corpus/). With the environment in place, we start where serious training starts: with the data.
