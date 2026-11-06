from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "models_catalog.json"
PROJECTS = ROOT / "projects"
STATIC = Path(__file__).resolve().parent / "static"
THIRD_PARTY = ROOT / "third_party"
EFFICIENTTAM_REPO = THIRD_PARTY / "EfficientTAM"
