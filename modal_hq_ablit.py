"""
HyperQwen fork serving an ABLITERATED AutoRound checkpoint on Modal.

Julio's pick (2026-10-28): the syv-ai/HyperQwen patched-vLLM stack
(the one that runs Qwen3.8-27B at ~110-125 tok/s on an RTX 3090),
but fed with an abliterated checkpoint instead of the base model:
ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound — a
compressed-tensors pack-quantized export that has ALREADY been
through this fork's preparation: its index carries lm_head /
embed / MTP in packed form and it ships mtp_draft_vocab_ids.pt.
(Two neighbouring repos do not work: letechlead's is a native
auto-round export, which the fork's prepare scripts explicitly
refuse — they extend a compressed-tensors config, they do not
convert one.) The fork's steps, for reference:

    quant_lm_head.py   lm_head  -> int8   (the step whose absence
    quant_embed.py     embed    -> int8    makes stock-vLLM MTP OOM:
    quant_mtp.py       MTP      -> int8    a bf16 draft lm_head is
    build_draft_vocab  40k draft head      2.37 GiB on a 22 GiB card)

GPU: A10 (sm86 — the fork's tuning target is the 3090, same arch;
Modal has no 3090 SKU, and an A10 exposes the same 22.1 GiB as an
L4, which is precisely the memory geometry this stack is tuned
for). MAX_LEN is the model's native 262,144 since 2026-10-29
(CTX=huge/KVarN; it was 32768 under CTX=fast at first deployment).

Cost guards, as everywhere today: max_containers=1,
scaledown_window=120. Secrets: reuses the `qwen38-nvfp4` Modal
secret (VLLM_API_KEY), so the abliterated endpoint shares the key
Julio already has. Served model name is the fork's: qwen3.8-27b.
"""

import json
import os
import subprocess
import time
import urllib.request

import modal

APP_NAME = "qwen38-hq-ablit"
HF_REPO = "ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound"
MODEL_DIR = "/app/models/Qwen3.8-27B-W4A16-AutoRound-ABLITERATED"
SERVED_NAME = "qwen3.8-27b"
PORT = 18020

app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name("qwen38-hq-ablit-models",
                                       create_if_missing=True)
cache_volume = modal.Volume.from_name("qwen38-hq-ablit-cache",
                                      create_if_missing=True)

image = (
    modal.Image.from_registry("ghcr.io/syv-ai/hyperqwen:sha-ce9615c",
                              add_python="3.12")
    .entrypoint([])
)

PREPARE_ENV = {
    "HF_REPO": HF_REPO,
    "BASE_MODEL_DIR": MODEL_DIR,
    "FAST_VARIANT": "0",   # its builder hardlinks; Modal Volumes refuse
    "DFLASH2": "0",        # SPEC=mtp only for this deployment
    "HOME": "/hq-cache",
}

SERVE_ENV = {
    "SPEC": "mtp",
    # CTX=huge (Julio, 2026-10-29): the model's native context,
    # 262,144 tokens, via the fork's KVarN 4/2-bit KV cache. CTX=fast
    # (bf16 KV) physically cannot hold 262k on a 22.1 GiB card.
    # Trade-off (fork docs + our bench): a few tok/s at chat length,
    # roughly half decode rate past 100k of context depth, and the
    # KV cache is lossy-compressed instead of exact bf16.
    "CTX": "huge",
    # 262,144 tokens of KVarN KV needs 3.89 GiB; at the fork's
    # default GPU_UTIL=0.93 only 2.87 GiB is left after weights,
    # drafts and graphs on the 22.1 GiB card, and vLLM refuses to
    # boot ("estimated maximum model length" ~193k). 0.98 leaves
    # ~3.98 GiB — enough for exactly one full-length sequence;
    # short chats share the pool as usual.
    "GPU_UTIL": "0.98",
    "HOST": "0.0.0.0",
    "PORT": str(PORT),
    "MODEL": MODEL_DIR,
    "BASE_MODEL_DIR": MODEL_DIR,
    "FAST_VARIANT": "0",
    "DFLASH2": "0",
    "PREPARE": "0",        # prepare() below already did the work
    "MAX_LEN": "262144",
    "HOME": "/hq-cache",
}


@app.function(image=image, volumes={"/app/models": models_volume,
                                    "/hq-cache": cache_volume},
              cpu=8, memory=32768, timeout=3600, env=PREPARE_ENV)
def prepare():
    """Download the abliterated checkpoint and run the fork's
    prepare over it (CPU-only; for this already-prepared export the
    quant steps all detect their markers and skip)."""
    marker = os.path.join(MODEL_DIR, ".source_repo")
    try:
        with open(marker) as f:
            current = f.read().strip()
    except OSError:
        current = None
    if current != HF_REPO:
        import shutil
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
    r = subprocess.run(["bash", "/app/docker/entrypoint.sh", "prepare"],
                       capture_output=True, text=True, timeout=3300)
    print(r.stdout[-6000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"prepare failed rc={r.returncode}")
    with open(marker, "w") as f:
        f.write(HF_REPO)
    models_volume.commit()
    cache_volume.commit()


@app.function(
    image=image,
    gpu="A10",
    cpu=4,
    memory=16384,
    volumes={"/app/models": models_volume, "/hq-cache": cache_volume},
    secrets=[modal.Secret.from_name("qwen38-nvfp4")],
    env=SERVE_ENV,
    timeout=86400,
    startup_timeout=3600,
    max_containers=1,
    scaledown_window=120,
)
@modal.web_server(PORT, startup_timeout=3600)
def serve():
    """Popen + return: an exec'd server listens but Modal never
    routes to it (learned on the HyperQwen deployment, 2026-10-28)."""
    env = dict(os.environ)
    env.update(SERVE_ENV)
    subprocess.Popen(["bash", "/app/docker/entrypoint.sh", "single"],
                     env=env)


CPU_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.34")
)


@app.function(image=CPU_IMAGE, cpu=1, memory=1024, timeout=1800,
              secrets=[modal.Secret.from_name("qwen38-nvfp4")])
def bench(url: str, tokens: int = 512):
    """Wait for health, warm up, then time forced-length generations
    (min_tokens + ignore_eos) and report completion tok/s."""
    key = os.environ["VLLM_API_KEY"]
    deadline = time.time() + 1500
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                url + "/health",
                headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(20)
    else:
        return {"error": "health deadline exceeded"}

    def one_gen():
        body = json.dumps({
            "model": SERVED_NAME,
            "messages": [{"role": "user", "content":
                          "Write a long, detailed essay about the "
                          "history of maritime trade in the "
                          "Mediterranean. Keep writing until told "
                          "to stop."}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": tokens, "min_tokens": tokens,
            "ignore_eos": True, "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            url + "/v1/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
        dt = time.time() - t0
        n = out["usage"]["completion_tokens"]
        return {"tokens": n, "seconds": round(dt, 2),
                "tok_per_s": round(n / dt, 1)}

    one_gen()  # warm-up
    runs = [one_gen(), one_gen()]
    best = max(runs, key=lambda x: x["tok_per_s"])
    return {"runs": runs, "best_tok_per_s": best["tok_per_s"]}


@app.local_entrypoint()
def do_bench(url: str):
    print(bench.remote(url))
