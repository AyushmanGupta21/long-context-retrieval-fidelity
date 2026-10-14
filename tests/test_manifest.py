"""
Validation tests for corpus manifest and documents on disk.
"""

import hashlib
import json
from pathlib import Path
import pytest
from src.parser import count_words

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.json"
DOCS_DIR = PROJECT_ROOT / "data" / "documents"


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="corpus_manifest.json not present")
def test_corpus_manifest_structure():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_documents"] == 10
    assert manifest["validated_documents"] == 10
    assert manifest["failed_documents"] == 0
    assert len(manifest["documents"]) == 10

    for doc in manifest["documents"]:
        assert doc["status"] == "validated"
        assert doc["actual_word_count"] >= 5000
        assert doc["sha256"] is not None
        assert doc["local_filename"].endswith(".txt")


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="corpus_manifest.json not present")
def test_corpus_documents_on_disk():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for doc in manifest["documents"]:
        file_path = DOCS_DIR / doc["local_filename"]
        assert file_path.exists(), f"Missing file: {file_path}"
        text = file_path.read_text(encoding="utf-8")
        actual_words = count_words(text)
        assert actual_words == doc["actual_word_count"]
        assert actual_words >= 5000
        # Verify sha256 checksum
        computed_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert computed_sha == doc["sha256"]
