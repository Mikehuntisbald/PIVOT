import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools import build_stageb_data_driven_role_routed_clean_assignment as builder


class RoleRoutedCleanAssignmentBuilderTest(unittest.TestCase):
    def test_base_row_stream_matches_upstream_per_row_digest_contract(self):
        row = {"z": [2, 1], "a": "value"}
        canonical = json.dumps(
            row,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        expected = hashlib.sha256(canonical).hexdigest().encode("ascii") + b"\n"
        self.assertEqual(builder._base_row_sha256_line(row), expected)
        self.assertNotEqual(
            hashlib.sha256(builder._base_row_sha256_line(row)).hexdigest(),
            hashlib.sha256(canonical + b"\n").hexdigest(),
        )

    def test_atomic_publish_never_replaces_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            destination = root / "published"
            first.mkdir()
            (first / "sentinel").write_text("first", encoding="ascii")
            builder._rename_directory_noreplace(first, destination)
            self.assertFalse(first.exists())
            self.assertEqual(
                (destination / "sentinel").read_text(encoding="ascii"), "first"
            )

            second = root / "second"
            second.mkdir()
            (second / "sentinel").write_text("second", encoding="ascii")
            with self.assertRaisesRegex(
                builder.CleanAssignmentBuildError, "refusing concurrent overwrite"
            ):
                builder._rename_directory_noreplace(second, destination)
            self.assertEqual(
                (destination / "sentinel").read_text(encoding="ascii"), "first"
            )
            self.assertEqual(
                (second / "sentinel").read_text(encoding="ascii"), "second"
            )

    def test_file_record_and_verify_reject_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "regular"
            regular.write_text("payload", encoding="ascii")
            link = root / "link"
            link.symlink_to(regular)
            with self.assertRaisesRegex(
                builder.CleanAssignmentBuildError, "symlinks are forbidden"
            ):
                builder._file_record(link)

            output = root / "output"
            output.mkdir()
            output_link = root / "output-link"
            output_link.symlink_to(output, target_is_directory=True)
            with self.assertRaisesRegex(
                builder.CleanAssignmentBuildError,
                "output root must not be a symlink",
            ):
                builder.verify(output_root=output_link)

    def test_sealed_assets_dry_run_replays_all_preregistered_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "must-remain-absent"
            plan = builder.make_plan(output_root=output_root)
            self.assertFalse(output_root.exists())
        self.assertEqual(plan.receipt["rows"], 263661)
        self.assertEqual(plan.receipt["valid_rows"], 224723)
        self.assertEqual(plan.receipt["invalid_rows"], 38938)
        self.assertEqual(plan.receipt["unique_image_keys"], 22359)
        for name, expected in builder.EXPECTED_OUTPUT.items():
            observed = plan.receipt["manifests"][name]
            self.assertEqual(observed["rows"], expected["rows"])
            self.assertEqual(observed["valid_rows"], expected["valid_rows"])
            self.assertEqual(observed["invalid_rows"], expected["invalid_rows"])
            self.assertEqual(observed["output"]["sha256"], expected["sha256"])
            self.assertEqual(
                observed["base_row_stream_sha256"],
                expected["base_row_stream_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
