import subprocess

import pytest

from tests.benchmark.attestation import (
    _canonical,
    _compact_report,
    _openssl_sign,
    _openssl_verify,
)


def test_rsa_signature_roundtrip_and_tamper_detection(tmp_path):
    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
    subprocess.run(
        [
            "openssl",
            "genpkey",
            "-algorithm",
            "RSA",
            "-pkeyopt",
            "rsa_keygen_bits:2048",
            "-out",
            str(private_key),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "openssl",
            "pkey",
            "-in",
            str(private_key),
            "-pubout",
            "-out",
            str(public_key),
        ],
        check=True,
        capture_output=True,
    )
    payload = _canonical({"tested_commit": "abc", "score": 0.97})
    signature = _openssl_sign(payload, private_key)
    _openssl_verify(payload, signature, public_key)
    with pytest.raises(RuntimeError):
        _openssl_verify(payload + b"tampered", signature, public_key)


def test_compact_report_drops_records_and_evidence_text():
    compact = _compact_report(
        {
            "dataset": {"split": "blind_test", "questions": 111},
            "summary": {"retrieval": {"recall_at_10": 0.97}},
            "smoke_gate": {"success": True, "results": ["private detail"]},
            "provenance": {
                "dataset_sha256": "dataset",
                "git": {"commit": "abc", "dirty": False},
                "model_fingerprint": "model",
                "threshold_fingerprint": "threshold",
            },
            "records": [{"question": "not committed", "evidence": "not committed"}],
        }
    )
    assert "records" not in compact
    assert compact["smoke_gate"] == {"success": True}
    assert "private detail" not in str(compact)
