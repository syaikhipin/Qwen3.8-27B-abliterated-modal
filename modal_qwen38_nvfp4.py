"""
Qwen3.8-27B abliterated NVFP4 on Modal — plain vLLM serving.

Model: Vtuber-plan/Huihui-Qwen3.8-27B-abliterated-NVFP4 (Huihui
abliteration of Qwen3.8-27B, ModelOpt NVFP4 export, VL checkpoint,
~19.2 GiB of weights). Julio asked for this exact repo on 2026-10-28
after the repo he named (vllm/Qwen3.8-27B-abliterated-NVFP4) turned
out not to exist on HuggingFace.

Design (carries the lessons from the HyperQwen deployment, same day):
  * Official vLLM image pinned by tag (vllm/vllm-openai:v0.31.0) —
    same vLLM generation that serves Qwen3.8 in the HyperQwen stack.
    NVFP4 on the L4 (sm89, no native FP4 units) runs through vLLM's
    Marlin dequant path; expect a speed penalty vs Blackwell.
  * serve() launches `vllm serve` with subprocess.Popen and RETURNS.
    Never exec: an exec'd server listens but Modal's web_server
    wrapper never routes requests to it (learned the hard way).
  * GPU: L4 at Julio's request. Modal exposes only ~22.1 GiB of the
    card, and the weights are ~19.2 GiB, so context is capped at
    16,384 (the hybrid architecture helps: only 16 of 64 layers are
    full attention, ~64 KiB KV per token) and gpu_memory_utilization
    is 0.97. If a future boot OOMs, the levers, in order, are
    MAX_MODEL_LEN 8192, then ENFORCE_EAGER=1 (drops CUDA graphs).
  * Free-plan cost guards (Julio, 2026-10-28): max_containers=1,
    scaledown_window=120 — a cold-start polling storm must never
    spawn a fleet again.
  * HF weights and the vLLM compile cache live on one volume
    (qwen38-nvfp4-hf) so scale-to-zero restarts skip the download.
  * The OpenAI API key comes from the Modal secret `qwen38-nvfp4`
    (VLLM_API_KEY in the serve env, --api-key for vLLM).

Runbook: see README.md in this directory.
"""

import os
import subprocess
import time
import urllib.request

import modal

APP_NAME = "qwen38-nvfp4"
REPO_ID = "Vtuber-plan/Huihui-Qwen3.8-27B-abliterated-NVFP4"
MODEL_DIR = "/hf-cache/models/Huihui-Qwen3.8-27B-abliterated-NVFP4"
SERVED_NAME = "qwen3.8-27b-abliterated-nvfp4"
PORT = 8000
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 16
GPU_MEM_UTIL = 0.98

app = modal.App(APP_NAME)

hf_volume = modal.Volume.from_name("qwen38-nvfp4-hf", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0", add_python="3.12")
    .entrypoint([])
    .env({"HF_HOME": "/hf-cache"})
)

prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.34")
    .env({"HF_HOME": "/hf-cache"})
)


@app.function(image=prep_image, volumes={"/hf-cache": hf_volume},
              cpu=4, memory=8192, timeout=1800)
def prepare():
    """One-time download of the checkpoint into the volume (~20.6 GB)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=REPO_ID, local_dir=MODEL_DIR)
    hf_volume.commit()
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print(f"prepare: {REPO_ID} at {path}, {total/1e9:.2f} GB")


@app.function(
    image=vllm_image,
    gpu="L4",
    cpu=4,
    memory=16384,
    volumes={"/hf-cache": hf_volume},
    secrets=[modal.Secret.from_name("qwen38-nvfp4")],
    env={"ENFORCE_EAGER": "1"},
    timeout=86400,
    startup_timeout=3600,
    max_containers=1,
    scaledown_window=120,
)
@modal.web_server(PORT, startup_timeout=3600)
def serve():
    """vLLM OpenAI server. Popen + return (see module docstring)."""
    args = [
        "vllm", "serve", MODEL_DIR,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        # Text-only serving: with image/video disabled vLLM skips the
        # vision-encoder dummy profiling, whose activation peak ate
        # ~1-2 GiB the 22.1 GiB card does not have (the 8k-context
        # boot died with 0.32 GiB KV available vs 0.57 GiB needed).
        # The tower weights still load; re-enable in the args if
        # vision input is ever wanted AND memory allows.
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
    ]
    if os.environ.get("ENFORCE_EAGER") == "1":
        args.append("--enforce-eager")
    env = dict(os.environ)
    env.setdefault("VLLM_CACHE_ROOT", "/hf-cache/vllm-cache")
    subprocess.Popen(args, env=env)


@app.function(image=prep_image, cpu=1, memory=1024, timeout=1800,
              secrets=[modal.Secret.from_name("qwen38-nvfp4")])
def ping(url: str):
    """Server-side smoke test: poll /health, then one chat completion."""
    key = os.environ["VLLM_API_KEY"]
    deadline = time.time() + 1500
    health = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url + "/health",
                                         headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                health = r.status
            break
        except Exception as e:  # noqa: BLE001 - report and retry
            print(f"ping: health not ready ({e}); retrying")
            time.sleep(20)
    if health != 200:
        return {"health": health, "error": "health deadline exceeded"}
    import json
    body = json.dumps({
        "model": SERVED_NAME,
        "messages": [{"role": "user",
                      "content": "Say hello in one short sentence."}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 64,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return {"health": 200,
            "reply": out["choices"][0]["message"]["content"],
            "usage": out.get("usage")}


@app.local_entrypoint()
def do_ping(url: str):
    print(ping.remote(url))
