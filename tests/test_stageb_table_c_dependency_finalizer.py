from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_stageb_table_c_dependency_closure import SyntheticTableC
from tools import audit_stageb_table_c_dependency_closure as audit
from tools import finalize_stageb_table_c_dependency_closure as finalizer


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def _complete_remaining_queue(fixture: SyntheticTableC) -> None:
    queue_path = fixture.remaining_queue / "queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["status"] = "completed"
    for item in queue["items"]:
        item["status"] = "completed"
    _write_json(queue_path, queue)


def _temporary_artifacts(output: Path) -> list[Path]:
    return sorted(output.parent.glob(f".{output.name}.*"))


class StageBTableCDependencyFinalizerTest(unittest.TestCase):
    def _prepared_fixture(
        self, root: Path
    ) -> tuple[SyntheticTableC, Path, dict[str, object]]:
        fixture = SyntheticTableC(root)
        preflight = fixture.create()
        _complete_remaining_queue(fixture)
        output = root / "final.json"
        return fixture, output, preflight

    def test_finalize_publishes_exact_completed_upgrade_and_source_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture, output, preflight = self._prepared_fixture(Path(temporary))
            with fixture.queue_verifier():
                result = finalizer.finalize_attestation(
                    output,
                    preflight_path=fixture.output,
                )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["publication_status"], "published")
            self.assertEqual(result["attestation"], str(output.resolve()))
            self.assertTrue(output.is_file())
            self.assertEqual(_temporary_artifacts(output), [])

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], audit.SCHEMA)
            self.assertEqual(payload["auditor_sources"], preflight["auditor_sources"])
            self.assertEqual(
                payload["queues"]["completed_l0_l4"]["completion_verification"]
                ["verified_item_count"],
                5,
            )
            self.assertEqual(
                payload["queues"]["remaining_table_c"]["completion_verification"]
                ["verified_item_count"],
                28,
            )
            self.assertEqual(
                payload["queues"]["remaining_table_c"]["status_policy"],
                "completed_required",
            )
            lineage = payload["finalization"]
            self.assertEqual(lineage["policy"], "final")
            self.assertEqual(
                lineage["preflight"]["semantic_attestation_sha256"],
                preflight["attestation_sha256"],
            )
            self.assertEqual(
                Path(lineage["finalizer_source"]["path"]).resolve(),
                Path(finalizer.__file__).resolve(),
            )
            self.assertTrue(
                lineage["auditor_preservation"]
                ["historical_auditor_sources_unchanged"]
            )

            with fixture.queue_verifier():
                replay = finalizer.verify_final_attestation(
                    output,
                    preflight_path=fixture.output,
                )
            self.assertEqual(replay["status"], "passed")
            self.assertTrue(replay["staged_upgrade_replayed"])

    def test_old_verifier_accepts_policy_tamper_but_final_verifier_rejects_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture, output, _ = self._prepared_fixture(Path(temporary))
            with fixture.queue_verifier():
                finalizer.finalize_attestation(
                    output,
                    preflight_path=fixture.output,
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            payload["queues"]["remaining_table_c"]["status_policy"] = (
                "running_or_completed"
            )
            payload["attestation_sha256"] = audit._attestation_digest(payload)
            _write_json(output, payload)

            with fixture.queue_verifier():
                old_result = audit.verify_attestation(
                    output,
                    policy="final",
                    config_entries=fixture.config_entries,
                )
            self.assertEqual(old_result["status"], "passed")
            with fixture.queue_verifier(), self.assertRaisesRegex(
                finalizer.TableCFinalizationError,
                "exact completed-policy upgrade",
            ):
                finalizer.verify_final_attestation(
                    output,
                    preflight_path=fixture.output,
                )

    def test_final_verifier_rejects_auditor_and_finalizer_source_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture, output, _ = self._prepared_fixture(Path(temporary))
            with fixture.queue_verifier():
                finalizer.finalize_attestation(
                    output,
                    preflight_path=fixture.output,
                )
            original = json.loads(output.read_text(encoding="utf-8"))

            cases: list[tuple[str, dict[str, object], str]] = []
            auditor_tamper = json.loads(json.dumps(original))
            auditor_tamper["auditor_sources"].pop()
            auditor_tamper["attestation_sha256"] = audit._attestation_digest(
                auditor_tamper
            )
            cases.append(("auditor", auditor_tamper, "auditor_sources"))

            finalizer_tamper = json.loads(json.dumps(original))
            finalizer_tamper["finalization"]["finalizer_source"]["sha256"] = "0" * 64
            finalizer_tamper["attestation_sha256"] = audit._attestation_digest(
                finalizer_tamper
            )
            cases.append(("finalizer", finalizer_tamper, "finalizer source"))

            for name, tampered, message in cases:
                with self.subTest(name=name):
                    _write_json(output, tampered)
                    with fixture.queue_verifier():
                        old_result = audit.verify_attestation(
                            output,
                            policy="final",
                            config_entries=fixture.config_entries,
                        )
                    self.assertEqual(old_result["status"], "passed")
                    with fixture.queue_verifier(), self.assertRaisesRegex(
                        finalizer.TableCFinalizationError,
                        message,
                    ):
                        finalizer.verify_final_attestation(
                            output,
                            preflight_path=fixture.output,
                        )

    def test_final_verifier_rejects_changes_outside_declared_policy_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture, output, _ = self._prepared_fixture(Path(temporary))
            with fixture.queue_verifier():
                finalizer.finalize_attestation(
                    output,
                    preflight_path=fixture.output,
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            payload["created_at_utc"] = "2040-01-01T00:00:00+00:00"
            payload["attestation_sha256"] = audit._attestation_digest(payload)
            _write_json(output, payload)

            with fixture.queue_verifier():
                old_result = audit.verify_attestation(
                    output,
                    policy="final",
                    config_entries=fixture.config_entries,
                )
            self.assertEqual(old_result["status"], "passed")
            with fixture.queue_verifier(), self.assertRaisesRegex(
                finalizer.TableCFinalizationError,
                "one-field staged upgrade",
            ):
                finalizer.verify_final_attestation(
                    output,
                    preflight_path=fixture.output,
                )

    def test_finalize_is_fresh_only_and_atomic_no_replace_preserves_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture, output, _ = self._prepared_fixture(Path(temporary))
            output.write_text("existing\n", encoding="ascii")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                finalizer.finalize_attestation(
                    output,
                    preflight_path=fixture.output,
                )
            self.assertEqual(output.read_text(encoding="ascii"), "existing\n")
            self.assertEqual(_temporary_artifacts(output), [])

            source = Path(temporary) / "publish-source.json"
            destination = Path(temporary) / "publish-destination.json"
            source.write_text("candidate\n", encoding="ascii")
            destination.write_text("incumbent\n", encoding="ascii")
            with self.assertRaisesRegex(FileExistsError, "appeared concurrently"):
                finalizer._rename_noreplace(source, destination)
            self.assertEqual(source.read_text(encoding="ascii"), "candidate\n")
            self.assertEqual(
                destination.read_text(encoding="ascii"), "incumbent\n"
            )

    def test_finalize_removes_both_temporaries_when_candidate_verification_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture, output, _ = self._prepared_fixture(Path(temporary))
            with fixture.queue_verifier(), mock.patch.object(
                finalizer,
                "verify_final_attestation",
                side_effect=finalizer.TableCFinalizationError("injected failure"),
            ), self.assertRaisesRegex(
                finalizer.TableCFinalizationError, "injected failure"
            ):
                finalizer.finalize_attestation(
                    output,
                    preflight_path=fixture.output,
                )
            self.assertFalse(output.exists())
            self.assertEqual(_temporary_artifacts(output), [])


if __name__ == "__main__":
    unittest.main()
