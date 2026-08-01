"""Check that the documented repository assets are present and parseable."""

from pathlib import Path
import py_compile


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "src/data_pipeline.py",
    "src/model_training.py",
    "src/calibration.py",
    "src/threshold_optimizer.py",
    "src/explainability.py",
    "notebooks/01_EDA_and_Feature_Engineering.ipynb",
    "notebooks/02_Model_Training_and_Evaluation.ipynb",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    for source in (ROOT / "src").glob("*.py"):
        py_compile.compile(source, doraise=True)

    print(f"Repository check passed: {len(REQUIRED_FILES)} required artifacts available.")


if __name__ == "__main__":
    main()
