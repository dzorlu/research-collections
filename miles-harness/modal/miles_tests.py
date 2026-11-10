"""Run miles tests on Modal, in the miles dev image, against a local checkout.

    MILES_REPO=~/repos/miles-tito-session uvx modal run miles_tests.py \
        --cmd "pytest tests/fast/rollout/session/test_token_space_images.py tests/fast/utils/test_media_refs.py"
    MILES_REPO=... MILES_GPU=L4 uvx modal run miles_tests.py --cmd "pytest tests/fast-gpu/test_torchtitan_routing_replay.py"

Settings, all environment variables read when the script loads:
- ``MILES_REPO``: the checkout to test (default: the current directory). Uploaded on each run without ``.git``,
  so local edits run without a rebuild.
- ``MILES_GPU``: a GPU type (``T4``, ``L4``, ``A10G``, ``H100``...), or unset for CPU. The fast tests need none.
- ``MILES_IMAGE_TAG``: the ``radixark/miles`` tag (default below: the one GPU CI builds from). Modal pulls it once
  and caches it.
- ``MILES_TEST_TIMEOUT``: seconds before the run is killed (default 1800).
- ``HF_TOKEN``: forwarded when set (gated models).

Reused between runs: the image, and a Modal volume ``miles-hf-cache`` mounted at ``/hf-cache`` (``HF_HOME``),
so a model's files download once.
"""

import os
import subprocess

import modal

IMAGE_TAG = os.environ.get("MILES_IMAGE_TAG", "dev-202611101346")
REPO = os.path.abspath(os.path.expanduser(os.environ.get("MILES_REPO", ".")))
GPU = os.environ.get("MILES_GPU") or None
REMOTE_REPO = "/workspace/miles"

app = modal.App("miles-tests")
hf_cache = modal.Volume.from_name("miles-hf-cache", create_if_missing=True)
image = modal.Image.from_registry(f"radixark/miles:{IMAGE_TAG}", add_python=None).add_local_dir(
    REPO, remote_path=REMOTE_REPO, ignore=[".git", "**/__pycache__", "**/.pytest_cache"]
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=int(os.environ.get("MILES_TEST_TIMEOUT", 1800)),
    volumes={"/hf-cache": hf_cache},  # not /root/.cache/huggingface: the image ships files there
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else [],
)
def run(cmd: str) -> int:
    """Run ``cmd`` in the uploaded checkout as CI does: the checkout first on PYTHONPATH, output streamed."""
    env = {
        **os.environ,
        "PYTHONPATH": f"{REMOTE_REPO}:{REMOTE_REPO}/3rdparty/miles-ext-kernels:" + os.environ.get("PYTHONPATH", ""),
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/hf-cache",
    }
    print(f"$ {cmd}\n  image radixark/miles:{IMAGE_TAG}, gpu {GPU or 'none'}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=REMOTE_REPO, env=env).returncode


@app.local_entrypoint()
def main(cmd: str):
    code = run.remote(cmd)
    print(f"exit code {code}")
    raise SystemExit(code)
