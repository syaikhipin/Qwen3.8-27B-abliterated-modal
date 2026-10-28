# Huihui-Qwen3.8-27B-abliterated on Modal

Serve the abliterated Qwen3.8-27B NVFP4 checkpoint on
[Modal](https://modal.com) with vLLM — an OpenAI-compatible endpoint
that scales to zero when idle.

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

- `modal_qwen38_nvfp4.py` — the whole deployment: `prepare`
  (one-time weight download to the `qwen38-nvfp4-hf` volume),
  `serve` (the vLLM web endpoint), `ping`/`do_ping` (smoke test).
- The API key lives only in the Modal secret `qwen38-nvfp4`. Rotate
  it with `modal secret create qwen38-nvfp4 VLLM_API_KEY=<new> --force`
  followed by a redeploy.

## Notes

- The model checkpoint is a community abliteration of Qwen3.8-27B;
  its own license and terms are the ones on its HuggingFace page.
- Deployed and verified end-to-end on 2026-10-28 (`/health` 200,
  chat completion 200).
