from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.stageb_profile_dependency_audit import (
    ProfileDependencyAuditError,
    recursive_local_python_dependencies,
)


def _write(root: Path, relative: str, source: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


class StageBProfileDependencyAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        _write(
            self.root,
            "entry.py",
            "from pkg import left\nfrom pkg import right\n",
        )
        _write(self.root, "pkg/__init__.py")
        _write(self.root, "pkg/left.py", "from pkg import shared\n")
        _write(
            self.root,
            "pkg/right.py",
            "from pkg import shared\nfrom pkg import right_leaf\n",
        )
        _write(self.root, "pkg/shared.py", "VALUE = 'shared'\n")
        _write(self.root, "pkg/right_leaf.py", "VALUE = 'right'\n")
        _write(self.root, "pkg/unreachable.py", "from pkg import shared\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def closure(self, **kwargs: object) -> list[str]:
        paths = recursive_local_python_dependencies(
            ["entry.py"],
            repository_root=self.root,
            **kwargs,
        )
        self.assertTrue(all(path.is_absolute() for path in paths))
        return [path.relative_to(self.root).as_posix() for path in paths]

    def test_returns_deterministic_sorted_absolute_closure(self) -> None:
        expected = [
            "entry.py",
            "pkg/__init__.py",
            "pkg/left.py",
            "pkg/right.py",
            "pkg/right_leaf.py",
            "pkg/shared.py",
        ]
        self.assertEqual(self.closure(), expected)
        self.assertEqual(self.closure(), expected)

    def test_prunes_only_the_declared_direct_edge(self) -> None:
        paths = self.closure(pruned_edges=[("pkg/left.py", "pkg/shared.py")])
        self.assertIn("pkg/shared.py", paths)
        self.assertIn("pkg/left.py", paths)

    def test_pruning_entry_edge_removes_only_its_unreachable_subtree(self) -> None:
        paths = self.closure(pruned_edges=[("entry.py", "pkg/left.py")])
        self.assertNotIn("pkg/left.py", paths)
        self.assertIn("pkg/shared.py", paths)
        self.assertIn("pkg/right.py", paths)

    def test_include_path_remains_a_root_when_an_incoming_edge_is_pruned(self) -> None:
        paths = self.closure(
            include_paths=["pkg/left.py"],
            pruned_edges=[("entry.py", "pkg/left.py")],
        )
        self.assertIn("pkg/left.py", paths)

    def test_rejects_unreachable_prune_source(self) -> None:
        with self.assertRaisesRegex(
            ProfileDependencyAuditError,
            "source is not reachable",
        ):
            self.closure(
                pruned_edges=[("pkg/unreachable.py", "pkg/shared.py")]
            )

        with self.assertRaisesRegex(
            ProfileDependencyAuditError,
            "unreachable after applying",
        ):
            self.closure(
                pruned_edges=[
                    ("entry.py", "pkg/left.py"),
                    ("pkg/left.py", "pkg/shared.py"),
                ]
            )

    def test_rejects_non_direct_or_stale_prune_edge(self) -> None:
        with self.assertRaisesRegex(
            ProfileDependencyAuditError,
            "not a direct resolved local import",
        ):
            self.closure(pruned_edges=[("entry.py", "pkg/shared.py")])

    def test_rejects_missing_and_non_file_prune_paths(self) -> None:
        cases = [
            (("missing.py", "pkg/shared.py"), "source is missing"),
            (("entry.py", "missing.py"), "target is missing"),
            (("pkg", "pkg/shared.py"), "source is not a file"),
            (("entry.py", "pkg"), "target is not a file"),
        ]
        for edge, pattern in cases:
            with self.subTest(edge=edge), self.assertRaisesRegex(
                ProfileDependencyAuditError,
                pattern,
            ):
                self.closure(pruned_edges=[edge])

    def test_rejects_malformed_prune_and_parse_failure(self) -> None:
        with self.assertRaisesRegex(
            ProfileDependencyAuditError,
            r"must be a \(source, target\) pair",
        ):
            self.closure(pruned_edges=[("entry.py",)])  # type: ignore[list-item]

        with self.assertRaisesRegex(
            ProfileDependencyAuditError,
            r"must be a \(source, target\) pair",
        ):
            self.closure(pruned_edges=["entry.py"])  # type: ignore[list-item]

        _write(self.root, "broken.py", "def invalid(:\n")
        with self.assertRaisesRegex(
            ProfileDependencyAuditError,
            "could not parse Python dependency",
        ):
            recursive_local_python_dependencies(
                ["broken.py"],
                repository_root=self.root,
            )


if __name__ == "__main__":
    unittest.main()
