"""Download every model weight listed in models_catalog.json into ./models.

Only needs `huggingface_hub`. Safe to re-run: finished files are skipped.

    python scripts/download_models.py                 # editor + weights-only groups (~6 GB, +7 GB if SAM 3 access)
    python scripts/download_models.py --list          # show the catalog and what is on disk
    python scripts/download_models.py --only edgetam,sam2.1-tiny
    python scripts/download_models.py --include-huge  # also the ~100 GB EfficientSAM3 variant dump

Gated models (SAM 3, SAM 3.1): request access on their Hugging Face page, run
`huggingface-cli login` once, then re-run this script.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "models_catalog.json"
REPORT = ROOT / "models" / "download_report.json"
DEFAULT_GROUPS = {"editor", "weights-only"}


def load_catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))["models"]


def on_disk(entry):
    if not entry.get("dir"):
        return False
    d = ROOT / entry["dir"]
    if entry.get("checkpoint"):
        return (d / entry["checkpoint"]).exists()
    # ignore .cache/ (hf_hub keeps partial downloads there)
    return d.exists() and any(
        p.is_file() and p.stat().st_size > 1_000_000 and ".cache" not in p.relative_to(d).parts for p in d.rglob("*")
    )


def download(entry, attempts=4):
    repos = entry["repo"] if isinstance(entry["repo"], list) else [entry["repo"]]
    for repo in repos:
        for k in range(attempts):
            try:
                snapshot_download(
                    repo_id=repo,
                    local_dir=ROOT / entry["dir"],
                    allow_patterns=entry.get("allow"),
                )
                break
            except (GatedRepoError, RepositoryNotFoundError):
                raise
            except Exception as ex:  # flaky connections: hf_hub resumes partial files on retry
                if k == attempts - 1:
                    raise
                print(f"    retry {k + 1}/{attempts - 1} after {type(ex).__name__}", flush=True)
                time.sleep(5 * (k + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", help="comma-separated model ids")
    ap.add_argument("--include-huge", action="store_true")
    args = ap.parse_args()

    catalog = load_catalog()
    if args.list:
        for e in catalog:
            mark = "OK " if on_disk(e) else "-- "
            print(f"{mark} {e['id']:<22} {e['group']:<13} {e['size_mb']:>7} MB  {e['name']}")
        return

    groups = DEFAULT_GROUPS | ({"huge"} if args.include_huge else set())
    only = set(args.only.split(",")) if args.only else None
    todo = [e for e in catalog if e.get("repo") and (e["id"] in only if only else e["group"] in groups)]

    results = {}
    for e in todo:
        print(f"\n=== {e['id']}  ({e['size_mb']} MB)  -> {e['dir']}", flush=True)
        t0 = time.time()
        try:
            download(e)
            results[e["id"]] = "ok"
            print(f"    done in {time.time() - t0:.0f}s", flush=True)
        except GatedRepoError:
            results[e["id"]] = "gated: request access on huggingface.co, then `huggingface-cli login`"
            print(f"    GATED - request access at https://huggingface.co/{e['repo']} and re-run", flush=True)
        except RepositoryNotFoundError:
            results[e["id"]] = "repo not found"
            print("    repo not found", flush=True)
        except Exception as ex:  # keep going: one failed model must not stop the rest
            results[e["id"]] = f"error: {type(ex).__name__}: {ex}"
            print(f"    ERROR {type(ex).__name__}: {ex}", flush=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k:<22} {v}")
    sys.exit(0 if all(v == "ok" or v.startswith("gated") for v in results.values()) else 1)


if __name__ == "__main__":
    main()
