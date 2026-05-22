# Running a Real Local AI Agent on DGX Spark · Qwen 3.6-35B via Atlas

I bought a DGX Spark to do real work: running serious local AI agents and training foundation models from scratch - not to run benchmarks. 

*(If you are curious about the training side of this hardware, check out [SageGPT](https://github.com/airawatraj/sage-gpt), my 7.5M parameter Sanskrit SLM trained entirely from scratch on this same machine).*

I previously [pushed a 120B Nemotron setup](https://github.com/airawatraj/dgx-spark-nemotron-super-agent) past the community benchmark records, but I hit a hard speed ceiling at ~24 TPS. I knew this hardware could deliver both deep smarts and blazing speed within its real single-node constraints. I went hunting for a setup that could do two things together:
1. Solve the logic puzzles people call "unsolvable" for local models.
2. Deliver the 100+ TPS speed the DGX Spark deserves.

`RedHatAI/Qwen3.6-35B-A3B-NVFP4` running on the **Atlas engine** turned out to be that gem.

This repo is a lightweight DGX Spark setup with the absolute minimum pieces needed to download the model, format the cache, launch the highly-optimized Atlas container, and reproduce the speed and smarts benchmarks locally.

> ⚠️ **Personal workstation setup. Not for enterprise use. Use at your own risk.**

---
## Requirement 

I needed a local model with both Smarts and Speed.

### Solving a puzzle the community said no local LLM could crack

I came across a reddit thread that claimed ["There's not a SINGLE local LLM which can solve this logic puzzle"](https://www.reddit.com/r/LocalLLaMA/comments/1mblq5g/theres_not_a_single_local_llm_which_can_solve/) - only a few local models could do it at the time of posting.

Running under the `Cogni-Brain` alias, this Qwen 3.6 setup solved it locally in under **30 seconds** due to the massive speed bump.

https://github.com/airawatraj/dgx-spark-qwen-super-agent/blob/d398051f3ce19b7a137b492a13750fa72c7b1cbf/assets/openwebui_logic_puzzle.mp4

<p align="center"><i>Cogni-Brain reasoning through the Albert-Bernard-Cheryl puzzle at 130+ TPS—solving it in <b>under 30 seconds</b>, rivaling the speed of frontier models.</i></p>


## Benchmark Results

### Custom Script Benchmark (Single-Stream Latency)

> These results were measured with the custom `benchmark_speed.py` script while Open WebUI and NemoHermes were active in the background.
> Unlike vLLM, the Atlas engine utilizes native NVFP4 compute kernels and MTP K=2 speculative decoding to achieve these speeds without bogging down the host.

| Metric | Result |
|---|---|
| Single session TPS (average) | **128.1 tok/s** |
| Peak TPS (single session) | **131.9 tok/s** |
| 4 concurrent sessions (Python-limited) | **76.4 tok/s** |
| Max context window | **130,753 tokens** |
| TTFT (steady state) | **74 ms** |

<p align="center">
  <img src="./assets/benchmark_test_1-3.png" width="600" alt="Benchmark Test 1-3">
</p>

<p align="center">
  <img src="./assets/benchmark_test_4-5.png" width="600" alt="Benchmark Test 4-5">
</p>

### Smarts Evaluation (Tool-Use Benchmark)

> Benchmarked using `benchmark_smarts.py` in short mode against the local Atlas endpoint.

**The unexpected win:** Moving from a 120B model to a 35B model often means sacrificing reasoning capability for speed. However, this Qwen 3.6-35B setup actually **outperformed** the 120B Nemotron model on agentic tasks. 

Where the 120B model scored 93/100 (struggling slightly with parameter precision), this 35B model scored a **perfect 100/100**, handling complex multi-step chains, strict unit conversions, and graceful error recovery flawlessly. Combined with a median turn time of just 0.8s, the deployability score jumped from 72 to an elite **96/100**.

| Metric | Result |
|---|---|
| Overall score | **100 / 100** |
| Rating | **★★★★★ Excellent** |
| Passed / Partial / Failed | **15 / 0 / 0** |
| Best categories | **Perfect across all 5 categories** |
| Responsiveness | **88 / 100** (median turn: 0.8s) |
| Deployability | **96 / 100** |

<p align="center">
  <img src="./assets/benchmark_smarts_1.png" width="600" alt="Tool-eval benchmark summary 1">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_2.png" width="600" alt="Tool-eval benchmark summary 2">
</p>

<p align="center">
  <img src="./assets/benchmark_smarts_3.png" width="600" alt="Tool-eval benchmark summary 3">
</p>

---

## Comparison With My Nemotron Run

This table tracks the performance leap achieved by migrating from the 120B Nemotron/vLLM stack to the 35B Qwen/Atlas stack on the exact same DGX Spark hardware.

## Comparison With My Nemotron Run

This table tracks the performance leap achieved by migrating from the 120B Nemotron/vLLM stack to the 35B Qwen/Atlas stack on the exact same DGX Spark hardware.

| Metric | Nemotron-120B (vLLM) | Qwen 3.6-35B (Atlas) |
|---|---:|---:|
| Single-session TPS | 24.1 tok/s | **128.1 tok/s** |
| Peak single-session TPS | 24.8 tok/s | **131.9 tok/s** |
| 4-session Python TPS | 53.9 tok/s | **76.4 tok/s** |
| Max working context | 130,753 tokens | **130,753 tokens** |
| Smarts score | 93 / 100 | **100 / 100** |

---

## Hardware & Architecture Advancements

- **NVIDIA DGX Spark** (GB10 Grace-Blackwell Superchip)
- **128 GB unified memory** (CPU + GPU shared)
- **Native NVFP4 Compute:** Unlike previous vLLM setups that fell back to Marlin dequantization on SM121, the Atlas engine utilizes native Rust/CUDA NVFP4 kernels for full hardware acceleration.
- **MTP Speculative Decoding:** Atlas utilizes Qwen's Multi-Token Prediction (MTP) heads (`--num-drafts 1` / K=2) to predict and verify multiple tokens per forward pass without needing a separate draft model.

---

## Quick Start

> ⚠️ **Warning:** The setup scripts disable system swap to prevent unified memory thrashing.

```bash
# 1. Verify prerequisites
bash setup/install.sh

# 2. Download the Qwen 3.6-35B model & map the Hub cache
bash setup/download_model.sh

# 3. Launch Atlas
bash docker/start.sh

# 4. Follow logs
docker logs -f atlas-qwen36
# Wait for "Speculative decoding: ENABLED" and "Listening on 0.0.0.0:8000"

## Benchmark It

```bash
# Single-stream speed and context test
uv run benchmark/benchmark_speed.py

# Tool-use capability benchmark
uv run benchmark/benchmark_smarts.py

# Full spark-arena-style throughput sweep
uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_arena.csv
```

## Repository Structure

```markdown
dgx-spark-qwen-super-agent/
├── README.md                    ← this file
├── CITATION.cff                 ← citation metadata
├── LICENSE                      ← MIT license
├── setup/
│   ├── install.sh               ← verify Docker, uv/uvx, and Hugging Face auth
│   └── download_model.sh        ← fetch model weights and build the refs/main bridge
├── docker/
│   ├── start.sh                 ← launch Atlas with optimized K=2 and 16GB SHM
│   ├── stop.sh                  ← stop and remove the container
│   └── status.sh                ← health check and metrics
├── benchmark/
│   ├── benchmark_speed.py       ← TPS, TTFT, concurrency, and context benchmark
│   ├── benchmark_smarts.py      ← tool-eval-bench wrapper for capability checks
│   └── benchmark_speed_arena.py ← long llama-benchy sweep for spark-arena-style runs
└── assets/                      ← screenshots
```

## Key Fixes Over Standard Docker Deployments

To hit peak throughput on the Atlas engine, several Docker-level and engine-level limits must be overridden. These are baked into start.sh:

| Bottleneck | Standard Config | Optimized Atlas Config |
|---|---|---|
| IPC Choking on batching | Docker default (64MB) | `--shm-size=16gb` |
| Speculative depth | K=1 | `--num-drafts 1` (K=2) |
| System swap thrashing | OS default | `vm.drop_caches=3` and swapoff |
| Downstream API breaks | `Qwen3.6-35B` | `--served-model-name Cogni-Brain` |
| Cache path mismatch | Direct dir mount | Hub-cache bridging with `refs/main` |


## Environment Overrides

You can override the defaults with environment variables before running start.sh:

```bash
export MODEL_ID=RedHatAI/Qwen3.6-35B-A3B-NVFP4
export SERVED_MODEL_NAME=Cogni-Brain
export MODEL_SLUG=RedHatAI_Qwen3.6-35B-A3B-NVFP4
export MODEL_REPO_DIR=models--RedHatAI--Qwen3.6-35B-A3B-NVFP4
export HF_MODELS_ROOT=$HOME/hf-models
export CONTAINER_NAME=atlas-qwen36
export ATLAS_IMAGE=avarok/atlas-gb10:latest
export ATLAS_PORT=8000
export GPU_MEMORY_UTILIZATION=0.88
export MAX_SEQ_LEN=131072
```
