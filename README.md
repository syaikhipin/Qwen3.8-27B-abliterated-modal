# Huihui-Qwen3.8-27B-abliterated on Modal

Serve the abliterated Qwen3.8-27B on [Modal](https://modal.com) —
an OpenAI-compatible endpoint that scales to zero when idle.

**Start with the optimized deployment** (`modal_hq_ablit.py`): the
syv-ai HyperQwen patched-vLLM stack serving an AutoRound build of
the abliteration at **69.2 tok/s** on an A10 — 5.4× the stock-vLLM
NVFP4 setup on the same GPU. The stock deployment
(`modal_qwen38_nvfp4.py`, 12.8 tok/s on A10 / 7.8 on L4) is kept
below for reference and as a fallback.

## Benchmarks (measured 2026-10-28)

Forced-length generations (512 tokens, greedy, thinking off,
best of two runs after warm-up; the `bench` function in each app):

| Stack | Checkpoint | GPU | tok/s |
|---|---|---|---|
| HyperQwen fork, MTP spec (k=4) + CUDA graphs | ababaka AutoRound (int8 lm_head/embed/MTP, 40k draft head) | A10 | **69.2** |
| Stock vLLM 0.31, eager, no speculation | Vtuber-plan NVFP4 (Marlin weight-only) | A10 | 12.8 |
| Stock vLLM 0.31, eager, no speculation | Vtuber-plan NVFP4 (Marlin weight-only) | L4 | 7.8 |

Why the gap: NVFP4 on sm86/sm89 has no native FP4 path in stock
vLLM, so it dequantizes through Marlin in eager mode; speculative
decoding is unavailable there in practice, because vLLM builds the
MTP draft with a separate bf16 lm_head (2.37 GiB) that OOMs next to
17.9 GiB of weights on a 22.1 GiB card, and CUDA-graph capture does
not fit either. The fork's preparation quantizes lm_head, embed
and MTP to int8 and builds a 40k-token draft head, which is what
makes MTP + graphs fit — with 3.87 GiB of KV cache to spare at
32k context (acceptance length 2.72 in the run above).

## Optimized deployment (HyperQwen fork)

| | |
|---|---|
| Model | [`ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound`](https://huggingface.co/ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound) — compressed-tensors export, already fork-prepared. (letechlead's native auto-round export does **not** work: the fork's prepare scripts extend a compressed-tensors config and refuse to convert one.) |
| Engine | HyperQwen image `ghcr.io/syv-ai/hyperqwen:sha-ce9615c` (patched vLLM 0.31.0) |
| GPU | Modal A10 (sm86 — the fork's 3090 tuning target; ~$1.10/hr while running, $0 idle) |
| Served model name | `qwen3.8-27b` |
| Context | 32,768 tokens, MTP speculative decoding, CTX=fast |
| Modal app | `qwen38-hq-ablit` |

```bash
modal run modal_hq_ablit.py::prepare     # download + verify (~17 GB, once)
modal deploy modal_hq_ablit.py
modal run modal_hq_ablit.py::do_bench \
  --url https://<your-workspace>--qwen38-hq-ablit-serve.us-east.modal.run
```

Auth reuses the `qwen38-nvfp4` Modal secret below (same API key).
The `-fast` model variant is not built: its builder hardlinks
files and Modal Volumes refuse hardlinks (~7% left on the table).

---

## Stock deployment (reference)

Serve the abliterated Qwen3.8-27B NVFP4 checkpoint with vLLM.

| | |
|---|---|
| Model | [`Vtuber-plan/Huihui-Qwen3.8-27B-abliterated-NVFP4`](https://huggingface.co/Vtuber-plan/Huihui-Qwen3.8-27B-abliterated-NVFP4) (Huihui abliteration of Qwen3.8-27B, ModelOpt NVFP4 export) |
| Engine | vLLM 0.31.0 (official `vllm/vllm-openai:v0.31.0` image) |
| GPU | Modal L4 (24 GB class), ~$0.80/hr while running, $0 idle |
| Served model name | `qwen3.8-27b-abliterated-nvfp4` |
| Context | 8,192 tokens (see *Why these settings* below) |
| Modal app | `qwen38-nvfp4` |

## Quickstart

Prerequisites: a Modal account and the Modal CLI, authenticated
(`modal token new`). The commands below use a checkout of this repo.

```bash
# 1. Create the API key secret (pick your own key value)
modal secret create qwen38-nvfp4 VLLM_API_KEY=<your-key>

# 2. Download the checkpoint (~20.6 GB) into a persistent volume — once
modal run modal_qwen38_nvfp4.py::prepare

# 3. Deploy the endpoint
modal deploy modal_qwen38_nvfp4.py
# → https://<your-workspace>--qwen38-nvfp4-serve.us-east.modal.run

# 4. Smoke-test it (server-side: polls /health, then one chat call)
modal run modal_qwen38_nvfp4.py::do_ping \
  --url https://<your-workspace>--qwen38-nvfp4-serve.us-east.modal.run
```

## Usage

```bash
curl -L https://<your-workspace>--qwen38-nvfp4-serve.us-east.modal.run/v1/chat/completions \
  -H "Authorization: Bearer <your-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-abliterated-nvfp4",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "max_tokens": 128
  }'
```

Any OpenAI client works the same way — set its base URL to
`https://<your-workspace>--qwen38-nvfp4-serve.us-east.modal.run/v1`
and the API key to your secret value. Tool calling is enabled
(`qwen3_coder` parser).

Two cold-start facts worth knowing:

- After the container scales to zero, the next request waits several
  minutes while the model loads (weights come from the volume, no
  re-download). If Modal has no L4 free, scheduling can wait longer.
- Long waits are answered with HTTP 303 redirects — clients must
  follow them (hence `curl -L`).

## Why these settings

The checkpoint's weights load to ~17.9 GiB and an L4 exposes only
~22.1 GiB, so the configuration is deliberately memory-lean. Each of
these was set after a boot failed without it (all verified live on
2026-10-28):

- `--max-model-len 8192`, `--max-num-seqs 16`,
  `--gpu-memory-utilization 0.98` — leaves 3.0 GiB of KV cache
  (42,325 tokens pooled, ~5× concurrency at full 8k context).
- `--enforce-eager` — CUDA-graph capture memory does not fit next to
  the weights; 16k context with graphs OOM-thrashed during startup.
- `--limit-mm-per-prompt '{"image": 0, "video": 0}'` — text-only.
  The vision tower's weights still load, but its startup profiling
  peak (~1–2 GiB of dummy-image activations) is what pushed earlier
  boots over the edge. Re-enable vision only on a bigger GPU.
- NVFP4 on the L4 (sm89, no native FP4 units) runs through vLLM's
  Marlin kernel as weight-only FP4 — correct output, slower than on
  Blackwell. Expect chat-grade, not datacenter, throughput.

## Cost guards

- `max_containers=1` — one GPU container, ever. (During the first
  bring-up of a similar deployment, parallel health polls during a
  cold start made the autoscaler boot ~10 GPUs at once. Never again.)
- `scaledown_window=120` — the container exits 2 minutes after the
  last request; idle time bills nothing.
- `modal app stop qwen38-nvfp4` turns the deployment fully off.

## Files

- `modal_hq_ablit.py` — the optimized deployment (HyperQwen fork):
  `prepare` (download the AutoRound checkpoint to the
  `qwen38-hq-ablit-models` volume + the fork's own prepare),
  `serve` (A10 web endpoint), `bench`/`do_bench` (tok/s harness).
- `modal_qwen38_nvfp4.py` — the stock deployment: `prepare`
  (one-time weight download to the `qwen38-nvfp4-hf` volume),
  `serve` (the vLLM web endpoint), `ping`/`do_ping` (smoke test),
  `bench`/`do_bench` (the same tok/s harness).
- The API key lives only in the Modal secret `qwen38-nvfp4`. Rotate
  it with `modal secret create qwen38-nvfp4 VLLM_API_KEY=<new> --force`
  followed by a redeploy.

## Notes

- The model checkpoint is a community abliteration of Qwen3.8-27B;
  its own license and terms are the ones on its HuggingFace page.
- Deployed and verified end-to-end on 2026-10-28 (`/health` 200,
  chat completion 200).
