# Qwen3.8-27B abliterated NVFP4 on Modal

OpenAI-compatible vLLM serving of
`Vtuber-plan/Huihui-Qwen3.8-27B-abliterated-NVFP4` (Huihui-abliterated
Qwen3.8-27B, ModelOpt NVFP4 export) on a Modal L4.
Deployed 2026-10-28 at Julio's request. (The repo he first named,
`vllm/Qwen3.8-27B-abliterated-NVFP4`, does not exist on HuggingFace;
he picked this one from the real candidates.)

- App: `qwen38-nvfp4` — `modal_qwen38_nvfp4.py`
- Endpoint: https://nur-arifin-akbar--qwen38-nvfp4-serve.us-east.modal.run
- Model name for clients: `qwen3.8-27b-abliterated-nvfp4`
- Auth: `Authorization: Bearer <VLLM_API_KEY>` — the key lives in the
  Modal secret `qwen38-nvfp4`. Rotate:
  `modal secret create qwen38-nvfp4 VLLM_API_KEY=<new> --force`
  then redeploy.
- GPU: L4 (~$0.80/hr list, per-second billing). Cost guards:
  max_containers=1, scaledown_window=120 (scales to zero; idle = $0).
- Config (verified live 2026-10-28): max_model_len 8192,
  max_num_seqs 16, gpu_memory_utilization 0.98, enforce-eager,
  text-only (--limit-mm-per-prompt image/video 0 — the vision
  encoder's profiling peak does not fit; tower weights still load).
  Tool calling on (qwen3_coder parser). NVFP4 on sm89 runs via
  vLLM's MarlinNvFp4 kernel (weight-only FP4). Weights load to
  17.91 GiB; 3.01 GiB KV free = 42,325 tokens, 5.17x concurrency
  at 8k context. Do not raise context/graphs without re-checking
  memory: 16k + CUDA graphs OOM-thrashed, and 8k with vision
  profiling died at KV sizing (0.32 GiB free vs 0.57 needed).
- Cold start: weights load from the `qwen38-nvfp4-hf` volume (no
  re-download); expect several minutes. Clients must follow Modal's
  303 redirects on long waits (curl -L).

Commands (Modal CLI: ~/workspace/.venvs/modal/bin/modal):

    modal deploy modal_qwen38_nvfp4.py            # (re)deploy
    modal run modal_qwen38_nvfp4.py::prepare      # re-download weights
    modal run modal_qwen38_nvfp4.py::do_ping --url https://nur-arifin-akbar--qwen38-nvfp4-serve.us-east.modal.run
    modal app stop qwen38-nvfp4                   # turn fully off
