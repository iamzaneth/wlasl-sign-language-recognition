from typing import Any
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

VIDEO_DIR = DATA_DIR / "videos"

ANNOTATION_FILE = DATA_DIR / "WLASL_v0.3.json"

def load_annotations(annotation_file: Path) -> list[dict[str, Any]]:
    """Load annotations from a JSON file."""
    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}. Please ensure the dataset is downloaded and placed in the correct directory."
        )

    with annotation_file.open("r", encoding="utf-8") as f:
        return json.load(f)

def inspect_annotation(annotation: dict[str, Any]) -> None:
    for key, value in annotation.items():
        print(f"Key: {key:<15}"
              f"Type: {type(value).__name__:<10}" 
              f"Sample Value: {str(value)[:100]}")
        
def main():
    annotations = load_annotations(ANNOTATION_FILE)

    if not annotations:
        print("No annotations found in the dataset.")
        return

    inspect_annotation(annotations[0])

if __name__ == "__main__":
    main()