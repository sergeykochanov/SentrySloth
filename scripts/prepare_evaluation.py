"""Materialize labeled synthetic Git histories without executing their source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from git import Actor, Repo

from sentrysloth.evaluation import EvaluationCase, EvaluationDataset


def prepare(cases_path: Path, output: Path) -> Path:
    definitions = json.loads(cases_path.read_text())
    cases = []
    for definition in definitions:
        repo_path = output / definition["case_id"]
        repo_path.mkdir(parents=True, exist_ok=False)
        repo = Repo.init(repo_path)
        revisions = []
        actor = Actor("Evaluation fixture", "fixture@example.invalid")
        for revision in ("before", "after"):
            files = {
                "README.md": "# Fixture contract\n\n"
                "Request fields are untrusted. require_admin enforces authorization. "
                "delete_account is a protected side effect. The form parser promises "
                "a memory limit even for unbounded streams.\n",
                **definition[revision],
            }
            for name, content in files.items():
                target = repo_path / name
                if not target.resolve().is_relative_to(repo_path.resolve()):
                    raise ValueError(f"Fixture path escapes repository: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            repo.index.add(list(files))
            commit = repo.index.commit(
                revision,
                author=actor,
                committer=actor,
                author_date="2026-01-01 00:00:00 +0000",
                commit_date="2026-01-01 00:00:00 +0000",
            )
            repo.create_tag(revision)
            revisions.append(commit.hexsha)
        cases.append(
            EvaluationCase(
                case_id=definition["case_id"],
                split=definition["split"],
                label=definition["label"],
                reason=definition["reason"],
                expected=definition.get("expected", []),
                repo=str(repo_path.resolve()),
                from_sha=revisions[0],
                to_sha=revisions[1],
            )
        )
    manifest = output / "dataset.json"
    manifest.write_text(EvaluationDataset(cases=cases).model_dump_json(indent=2))
    (output / "repos.txt").write_text("\n".join(case.repo for case in cases) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for generated repositories")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation" / "cases.json",
    )
    args = parser.parse_args()
    print(prepare(args.cases, args.output))
