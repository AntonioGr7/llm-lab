# RunPod setup — full walkthrough

> Companion to [Module 01](README.md). Time: ~20 minutes the first time, ~90 seconds every time after. Cost to complete this guide: ~$0.50 (one short A100 session to verify everything works).

This guide takes you from "I've never used RunPod" to "I can spin up an A100, SSH in from my laptop, run a training script, and shut it down" — which is the workflow every Part 3+ module assumes.

If you've done all this before, the TL;DR is at the bottom.

## What you'll have at the end

- A RunPod account with billing set up.
- An SSH keypair on your laptop, with the public half uploaded to RunPod once (auto-injects into every future pod).
- A 50 GB **network volume** that survives pod termination — your HuggingFace cache and checkpoints don't get re-downloaded every session.
- A reusable `~/.ssh/config` entry so connecting is one command: `ssh runpod`.
- A verified A100 80GB pod running PyTorch 2.x + CUDA 12, that you can shut down and re-create at will.

## Step 1 — Account and billing

1. Sign up at [runpod.io](https://runpod.io).
2. **Billing → Add payment method.** Add a card.
3. **Top up $25 to start.** RunPod is pay-as-you-go; the balance just buys you runtime. $25 covers a few modules with room to spare.

A100 80GB **community cloud** pricing as of writing: ~$1.20–1.80/hr. Secure cloud is ~30% more for SOC2 / reserved capacity — you don't need it for course work.

## Step 2 — Generate an SSH key on your laptop

Skip this if you already have `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`) and use it for GitHub etc.

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept the default path (~/.ssh/id_ed25519)
# Set a passphrase if you want — optional but recommended
```

Print the public key (the half you upload):

```bash
cat ~/.ssh/id_ed25519.pub
```

You should see one line starting `ssh-ed25519 AAAA...`. Copy the whole line.

> **Why ed25519 not RSA:** shorter, faster, more secure. RunPod accepts both; ed25519 is the modern default.

## Step 3 — Upload the public key to RunPod

This is the one-time setup that means you never have to mess with SSH keys per-pod.

1. RunPod console → click your avatar (top right) → **Settings**.
2. Scroll to **SSH Public Keys** → **Add SSH Public Key**.
3. Paste the line from `cat ~/.ssh/id_ed25519.pub`. Save.

From now on, every pod you create has your public key dropped into `/root/.ssh/authorized_keys` automatically. No per-pod copying.

## Step 4 — Create a network volume (persistent storage)

A network volume is a disk that lives outside any specific pod. You mount it at `/workspace`, and it persists when the pod is destroyed. This is where your repo, HuggingFace cache, and checkpoints should live.

Without a network volume, every fresh pod re-downloads multi-gigabyte model weights.

1. **Storage** (left sidebar) → **Network Volumes** → **+ New Network Volume**.
2. **Region:** pick one with A100 availability. **US-CA-2** and **EUR-IS-1** are reliable. Whichever region you pick, your future pods must be in that same region to mount this volume.
3. **Size:** 50 GB is enough for Parts 0–2 + Module 11's pretraining checkpoints. Bump to 100 GB if you plan to keep multiple post-training checkpoints. You can resize up later, not down.
4. **Name:** `llm-lab`.
5. Create. Cost: ~$0.07/GB/month → ~$3.50/mo for 50 GB. You're billed even when no pod is attached, so delete it when the course is done.

## Step 5 — Deploy your first pod

1. **Pods** (left sidebar) → **+ Deploy**.
2. **GPU:** filter by **A100 80GB**. Pick a card in the region of your network volume. Sort by price.
3. **Cloud type:** **Community Cloud** for everything in this course. Secure cloud is for production / compliance, not course work.
4. **Pod template:** **RunPod PyTorch 2.x** (pick the latest version with CUDA 12.x). This image has Python 3.11, PyTorch with CUDA, openssh-server, and HuggingFace libs preinstalled.
5. **Customize Deployment** (expand the section):
   - **Container disk:** 20 GB. (This is scratch space — wiped on pod termination. Big enough for pip installs.)
   - **Volume:** Select your `llm-lab` network volume. Mount path: `/workspace` (default).
   - **Expose TCP Ports:** make sure **22** (SSH) and **8888** (Jupyter, optional) are checked. The PyTorch template usually does this for you.
   - **Environment Variables:** leave as-is for now; you'll set `WANDB_API_KEY` and `HF_TOKEN` per-session via login commands.
6. **Deploy On-Demand.** (Spot pods are ~50% cheaper but can be preempted. Use spot once you trust your checkpointing — Module 11 onwards. For your first pod, on-demand keeps the variables low.)

The pod takes ~30–60 seconds to start. State goes `Provisioning → Running`.

## Step 6 — Get the SSH command and configure your laptop

Once the pod is **Running**, click it to open the pod detail page, then **Connect**.

You'll see two SSH options:

- **SSH over exposed TCP** (the one you want — real SSH, supports scp/rsync/port-forwarding). Looks like:
  ```
  ssh root@123.45.67.89 -p 12345 -i ~/.ssh/id_ed25519
  ```
- **Basic SSH** (RunPod's proxy through their CLI — fine as a fallback, but limited).

Copy the **SSH over exposed TCP** command. The IP and port are unique to this pod and change every time you create a new one.

### Add a reusable `~/.ssh/config` entry

So you don't paste the long command every time, add an entry to `~/.ssh/config` on your laptop:

```ssh-config
Host runpod
    HostName 123.45.67.89        # replace with your pod's IP
    Port 12345                   # replace with your pod's port
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no     # pod IPs change; skip the host-key prompt
    UserKnownHostsFile /dev/null # don't pollute known_hosts with one-off IPs
    ServerAliveInterval 60       # keep SSH alive during long training runs
```

Now connect with just:

```bash
ssh runpod
```

When you destroy this pod and create a new one, edit `HostName` and `Port` to match the new pod. That's the only thing that changes.

> **Tip:** keep one terminal tab with `ssh runpod` open for the pod's lifetime. If it disconnects (laptop sleep, network change), just re-run — the `ServerAliveInterval` setting helps but isn't bulletproof. For long training runs, launch under `tmux` or `nohup` on the pod so the run survives disconnects.

## Step 7 — First-time setup inside the pod

You're now SSH'd in as `root`. The repo isn't there yet. Clone it into the network volume so it persists:

```bash
cd /workspace
git clone https://github.com/<your-fork>/llm-lab.git   # or the upstream
cd llm-lab

# Point HuggingFace at the network volume so cached weights persist
echo 'export HF_HOME=/workspace/hf-cache' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache' >> ~/.bashrc
source ~/.bashrc

# Install course requirements (the PyTorch image already has torch+CUDA)
pip install -r requirements.txt

# Log in to HF and W&B (paste tokens from your accounts)
huggingface-cli login
wandb login
```

Verify the GPU is visible:

```bash
nvidia-smi          # should show one A100-SXM4-80GB at ~0% util
nvcc --version      # CUDA 12.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: 2.x.x True NVIDIA A100-SXM4-80GB (or A100 80GB PCIe)
```

## Step 8 — Run the torchrun sanity check

This is Module 01's verification that distributed training works end-to-end on your pod:

```bash
cd /workspace/llm-lab/part-0-mental-model/01-tools-and-environment
torchrun --nproc_per_node=1 hello.py
```

Expected output:

```
[rank 0 / world_size 1] hello from cuda:0
[rank 0] all-reduce check: tensor sum = 1.0 (expected 1.0) ✓
```

If you see that, everything works. You are ready to run any module in the course.

## Step 9 — Shut down properly

**This is the single most important habit for keeping cost down.**

When you're done with a session:

1. Push any code changes (`git push`) — the network volume keeps the repo, but pushing is cheap accountability.
2. Make sure checkpoints you care about are saved under `/workspace/` (the network volume), not under `/root/` (the container disk — wiped on termination).
3. On the RunPod console, click your pod → **Stop** (pauses billing for the GPU but keeps the container disk, cheaper to resume) **or** **Terminate** (destroys everything not on the network volume, $0/hr).

**Default to Terminate** at the end of a work session. Use Stop only if you're stepping away for <1 hour. RunPod bills container disk storage even when stopped; over a week of "stopped" pods this adds up.

When you come back: deploy a new pod attached to the same network volume, update the IP/port in your `~/.ssh/config`, and you're back where you left off in 90 seconds.

## Cost cheat sheet

| Action | Cost |
|---|---|
| Idle pod (running but doing nothing) | Full hourly rate — **always shut down** |
| Stopped pod | Container disk only (~$0.10/hr for 20 GB) — fine for <1 hour breaks |
| Network volume, no pod attached | ~$0.07/GB/month — keeps your data alive between pods |
| A100 80GB community on-demand | $1.20–1.80/hr |
| A100 80GB community spot | ~$0.60–0.90/hr — preemptible, fine once your checkpointing is solid |
| H100 80GB community on-demand | $2.50–3.50/hr — only needed for the FP8 demo |

The course as a whole budgets ~$50. If you blow past that, the culprit is almost always a pod left running overnight.

## Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `ssh: connect to host … port …: Connection refused` | Pod still provisioning, or you copied the IP/port from an old pod | Wait 30s; re-copy from the **Connect** panel |
| `Permission denied (publickey)` | Public key not uploaded to RunPod account, or wrong `IdentityFile` in ssh config | Re-check Step 3, and `ls -l ~/.ssh/id_ed25519` exists |
| HuggingFace re-downloading weights every session | `HF_HOME` not pointing at `/workspace/...` | Re-do the `export HF_HOME=...` in `~/.bashrc` and `source` it |
| Pod has no A100 available in your region | Community capacity fluctuates | Try a different region, or switch to A100 40GB (most modules still fit), or wait 30 minutes |
| `torch.cuda.is_available() == False` on the pod | You picked a non-PyTorch template by mistake | Terminate, redeploy with the **RunPod PyTorch 2.x** template |
| Network volume won't mount | Pod is in a different region than the volume | Volumes are region-locked. Match the region exactly. |
| SSH disconnects mid-training and the run dies | Training process was tied to the SSH session | Always launch long runs under `tmux new -s train` then `Ctrl-B D` to detach |

## TL;DR (returning users)

```bash
# Once, ever
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub   # paste into RunPod account settings → SSH Public Keys
# Create a 50 GB network volume in your preferred region

# Each session
# 1. RunPod console → Deploy → A100 80GB community + PyTorch 2.x template + attach network volume
# 2. Update HostName/Port in ~/.ssh/config under `Host runpod`
ssh runpod
cd /workspace/llm-lab && git pull
tmux new -s train
torchrun --nproc_per_node=1 <module>/train.py --config=<module>/configs/...
# Ctrl-B D to detach, then `exit` to leave ssh. Training continues.

# When done
# Console → Terminate pod (keep the network volume)
```

## Next

Back to [Module 01](README.md#the-torchrun-sanity-check) to finish the `torchrun` verification, then on to [Module 02 — The Corpus](../../part-1-data/02-the-corpus/).
