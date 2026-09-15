import hashlib
import io
import zipfile

import pytest

from scripts.download_models import download


def test_download_extracts_only_expected_model_and_verifies_hash(tmp_path, monkeypatch):
    payload = b"model-test-content"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../unexpected.txt", "never extract this")
        z.writestr("bundle/model.onnx", payload)
    from scripts import download_models as module

    monkeypatch.setattr(
        module, "urlopen", lambda *a, **k: io.BytesIO(archive.getvalue())
    )
    spec = {
        "source": "https://example.invalid/model.zip",
        "file": "model.onnx",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    download(tmp_path, "pose", spec)
    assert list(tmp_path.iterdir()) == [tmp_path / "model.onnx"]
    assert (tmp_path / "model.onnx").read_bytes() == payload
    (tmp_path / "model.onnx").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(tmp_path, "pose", spec)
