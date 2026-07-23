from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts.extractor import make_qr_png


def test_public_qr_is_generated_without_logo_assets():
    with TemporaryDirectory() as tmp:
        output = Path(tmp) / "qr.png"
        make_qr_png({"download_id": "CDE-TEST", "file_name": "sample.xlsx"}, output)
        image = Image.open(output)
        assert image.size[0] > 0 and image.size[1] > 0
        assert image.mode in {"RGB", "1", "L"}


def test_removed_logo_assets_are_not_present():
    project = Path(__file__).resolve().parents[1]
    assert not (project / "static" / "img" / "tma_logo.png").exists()
    assert not (project / "static" / "img" / "coat_of_arms.png").exists()
