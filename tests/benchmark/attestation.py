"""Create and verify a signed, compact benchmark attestation.

The expensive blind benchmark runs once on a clean source commit.  This tool
signs only the report summary and provenance.  A following commit may add the
attestation file; PR CI then verifies the signature, source commit binding and
regression decision without calling any external model.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.benchmark.benchmark import load_json
from tests.benchmark.regression_gate import run_gate

DEFAULT_REPORT = PROJECT_ROOT / "tests/benchmark/results/regression_report.json"
DEFAULT_PROFILES = PROJECT_ROOT / "tests/benchmark/regression_profiles.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests/benchmark/attestations/current.json"
DEFAULT_PRIVATE_KEY = PROJECT_ROOT / ".benchmark/attestation_private_key.pem"
DEFAULT_PUBLIC_KEY = PROJECT_ROOT / "tests/benchmark/attestation_public_key.pem"
ALLOWED_POST_TEST_FILES = {
    "tests/benchmark/attestations/current.json",
}
SCHEMA_VERSION = 1


def _run(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _git(*args: str, check: bool = True) -> str:
    return _run("git", *args, check=check)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _baseline_binding(
    profiles: Mapping[str, Any], profiles_path: Path = DEFAULT_PROFILES
) -> dict[str, Any]:
    binding: dict[str, Any] = {"profiles_sha256": _sha256(profiles_path)}
    for name, tier in (profiles.get("tiers") or {}).items():
        configured = PROJECT_ROOT / str(tier.get("baseline") or "")
        fallback_value = tier.get("fallback_baseline")
        fallback = PROJECT_ROOT / str(fallback_value) if fallback_value else None
        selected = configured if configured.exists() else fallback
        if selected is None or not selected.exists():
            raise FileNotFoundError(f"missing {name} baseline")
        binding[name] = {
            "path": str(selected.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(selected),
        }
    return binding


def _compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    provenance = report.get("provenance") or {}
    return {
        "dataset": report.get("dataset") or {},
        "summary": report.get("summary") or {},
        "smoke_gate": {"success": bool((report.get("smoke_gate") or {}).get("success"))},
        "provenance": {
            "dataset_sha256": provenance.get("dataset_sha256"),
            "git": provenance.get("git") or {},
            "model_fingerprint": provenance.get("model_fingerprint"),
            "threshold_fingerprint": provenance.get("threshold_fingerprint"),
        },
    }


def _openssl_sign(payload: bytes, private_key: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "payload.json"
        signature = Path(tmp) / "signature.bin"
        source.write_bytes(payload)
        _run(
            "openssl", "dgst", "-sha256", "-sign", str(private_key),
            "-out", str(signature), str(source)
        )
        return base64.b64encode(signature.read_bytes()).decode("ascii")


def _openssl_verify(payload: bytes, signature_b64: str, public_key: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "payload.json"
        signature = Path(tmp) / "signature.bin"
        source.write_bytes(payload)
        signature.write_bytes(base64.b64decode(signature_b64, validate=True))
        _run(
            "openssl", "dgst", "-sha256", "-verify", str(public_key),
            "-signature", str(signature), str(source)
        )


def create_attestation(
    report_path: Path,
    profiles_path: Path,
    private_key: Path,
    public_key: Path,
) -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise ValueError("run on a clean committed source tree")
    report = load_json(report_path)
    profiles = load_json(profiles_path)
    tested_commit = _git("rev-parse", "HEAD")
    compact = _compact_report(report)
    report_git = (compact.get("provenance") or {}).get("git") or {}
    if report_git.get("commit") != tested_commit or report_git.get("dirty") is not False:
        raise ValueError("benchmark report is not bound to the current clean commit")
    gate = run_gate(compact, profiles, ["quality", "performance"])
    if gate.get("decision") != "PASS":
        raise ValueError("benchmark report does not pass the current regression gate")
    dataset_path = PROJECT_ROOT / "tests/benchmark/splits/blind_test.json"
    dataset = compact.get("dataset") or {}
    if dataset.get("split") != "blind_test" or dataset.get("questions") != 111:
        raise ValueError("attestation requires the complete 111-question blind_test")
    if dataset.get("sha256") != _sha256(dataset_path):
        raise ValueError("report dataset hash does not match blind_test.json")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tested_commit": tested_commit,
        "source_tree_hash": _git("rev-parse", f"{tested_commit}^{{tree}}"),
        "public_key_sha256": _sha256(public_key),
        "baseline_binding": _baseline_binding(profiles, profiles_path),
        "report": compact,
        "gate": gate,
    }
    return {
        **payload,
        "signature": {
            "algorithm": "rsa-sha256",
            "value": _openssl_sign(_canonical(payload), private_key),
        },
    }


def verify_attestation(
    attestation_path: Path,
    profiles_path: Path,
    public_key: Path,
) -> dict[str, Any]:
    attestation = load_json(attestation_path)
    signature = attestation.get("signature") or {}
    if signature.get("algorithm") != "rsa-sha256":
        raise ValueError("unsupported attestation signature algorithm")
    payload = {key: value for key, value in attestation.items() if key != "signature"}
    _openssl_verify(_canonical(payload), str(signature.get("value") or ""), public_key)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported attestation schema")
    if payload.get("public_key_sha256") != _sha256(public_key):
        raise ValueError("public key fingerprint mismatch")
    tested_commit = str(payload.get("tested_commit") or "")
    if not tested_commit:
        raise ValueError("missing tested_commit")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", tested_commit, "HEAD"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode:
        raise ValueError("tested_commit is not an ancestor of HEAD")
    expected_tree = _git("rev-parse", f"{tested_commit}^{{tree}}")
    if payload.get("source_tree_hash") != expected_tree:
        raise ValueError("tested source tree fingerprint mismatch")
    changed = {
        line for line in _git("diff", "--name-only", f"{tested_commit}..HEAD").splitlines()
        if line
    }
    unexpected = changed - ALLOWED_POST_TEST_FILES
    if unexpected:
        raise ValueError(
            "code changed after benchmark: " + ", ".join(sorted(unexpected))
        )
    profiles = load_json(profiles_path)
    if payload.get("baseline_binding") != _baseline_binding(profiles, profiles_path):
        raise ValueError("baseline or regression profile changed after benchmark")
    report = payload.get("report") or {}
    dataset_path = PROJECT_ROOT / "tests/benchmark/splits/blind_test.json"
    if (report.get("dataset") or {}).get("sha256") != _sha256(dataset_path):
        raise ValueError("attested dataset is stale")
    gate = run_gate(report, profiles, ["quality", "performance"])
    if gate.get("decision") != "PASS":
        raise ValueError("attested benchmark regresses against the current baseline")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Signed offline benchmark attestation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    create.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    create.add_argument("--private-key", type=Path, default=DEFAULT_PRIVATE_KEY)
    create.add_argument("--public-key", type=Path, default=DEFAULT_PUBLIC_KEY)
    create.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--attestation", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    verify.add_argument("--public-key", type=Path, default=DEFAULT_PUBLIC_KEY)
    args = parser.parse_args()
    if args.command == "create":
        result = create_attestation(
            args.report, args.profiles, args.private_key, args.public_key
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[PASS] signed attestation: {args.output}")
    else:
        gate = verify_attestation(args.attestation, args.profiles, args.public_key)
        print(f"[{gate['decision']}] signed offline regression attestation")


if __name__ == "__main__":
    main()
