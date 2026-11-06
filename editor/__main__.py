"""Start Track Studio: python -m editor [--port 7860] [--no-browser]"""

import argparse
import os
import threading
import webbrowser

# never let transformers import TensorFlow/JAX if they happen to be installed (slow, noisy, unused)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import uvicorn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Track Studio running at {url}")
    uvicorn.run("editor.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
