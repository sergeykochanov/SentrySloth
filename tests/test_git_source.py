"""Tests for GitSource local mode and ref validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from sentrysloth.config import get_settings
from sentrysloth.sources.git import GitSource, GitSourceError


def _init_repo(path: Path) -> None:
    repo = Repo.init(path)
    readme = path / "README.md"
    readme.write_text("# test\n", encoding="utf-8")
    repo.index.add(["README.md"])
    repo.index.commit("init")


@pytest.mark.asyncio
async def test_ensure_cloned_opens_local_repo(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)

    settings = get_settings(clone_base_dir=tmp_path / "clones")
    source = GitSource(str(repo_dir), settings)
    local_path = await source.ensure_cloned()

    assert local_path == repo_dir
    assert source.repo.working_tree_dir == str(repo_dir)


@pytest.mark.asyncio
async def test_ensure_cloned_remote_clones_without_partial_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = get_settings(clone_base_dir=tmp_path / "clones")
    source = GitSource("https://github.com/example/project.git", settings)

    fake_repo_api = MagicMock()
    cloned_repo = MagicMock()
    fake_repo_api.clone_from = MagicMock(return_value=cloned_repo)
    monkeypatch.setattr("sentrysloth.sources.git.Repo", fake_repo_api)

    local_path = await source.ensure_cloned()

    assert local_path == source.local_path
    fake_repo_api.clone_from.assert_called_once()
    _args, kwargs = fake_repo_api.clone_from.call_args
    assert "multi_options" not in kwargs
    assert source.repo is cloned_repo


@pytest.mark.asyncio
async def test_ensure_cloned_reclones_when_existing_clone_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = get_settings(clone_base_dir=tmp_path / "clones")
    source = GitSource("https://github.com/example/project.git", settings)
    local = source.local_path
    (local / ".git").mkdir(parents=True)
    marker = local / "stale.txt"
    marker.write_text("stale", encoding="utf-8")

    fake_repo_api = MagicMock(side_effect=InvalidGitRepositoryError("bad repo"))
    recloned_repo = MagicMock()
    fake_repo_api.clone_from = MagicMock(return_value=recloned_repo)
    monkeypatch.setattr("sentrysloth.sources.git.Repo", fake_repo_api)

    local_path = await source.ensure_cloned()

    assert local_path == local
    assert fake_repo_api.clone_from.call_count == 1
    _args, kwargs = fake_repo_api.clone_from.call_args
    assert "multi_options" not in kwargs
    assert not marker.exists()
    assert source.repo is recloned_repo


@pytest.mark.asyncio
async def test_get_diff_raises_on_invalid_ref(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)

    settings = get_settings(clone_base_dir=tmp_path / "clones")
    source = GitSource(str(repo_dir), settings)
    await source.ensure_cloned()

    with pytest.raises(GitSourceError, match="Invalid ref"):
        await source.get_diff("missing-ref", "HEAD")


@pytest.mark.asyncio
async def test_list_tags_skips_non_commit_tags(tmp_path: Path):
    """Some repos have tags pointing to blob/tree objects; we should skip them."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)

    repo = Repo(repo_dir)
    repo.create_tag("v1.0")

    # Create a tag that points to a blob (not a commit).
    blob_sha = repo.git.rev_parse("HEAD:README.md")
    repo.git.update_ref("refs/tags/git-to-hg-map", blob_sha)

    # Another normal tag on the current commit (same commit is fine for this regression).
    repo.create_tag("v1.1")

    settings = get_settings(clone_base_dir=tmp_path / "clones")
    source = GitSource(str(repo_dir), settings)
    await source.ensure_cloned()

    tags = await source.list_tags(limit=50)
    assert "git-to-hg-map" not in tags
    assert "v1.0" in tags
    assert "v1.1" in tags

    tags_with_dates = await source.list_tags_with_dates(limit=50)
    names = [t[0] for t in tags_with_dates]
    assert "git-to-hg-map" not in names
    assert "v1.0" in names
    assert "v1.1" in names


class TestPathTraversalProtection:
    """Tests for _is_safe_file_path validation."""

    @pytest.mark.parametrize(
        "path",
        [
            "src/main.py",
            "README.md",
            "a/b/c/d.txt",
            ".hidden/file.py",
            "dir/file-name.txt",
        ],
    )
    def test_safe_paths_accepted(self, path: str):
        assert GitSource._is_safe_file_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "../../etc/passwd",
            "../secret",
            "src/../../etc/shadow",
            "foo/../../../bar",
        ],
    )
    def test_traversal_rejected(self, path: str):
        assert GitSource._is_safe_file_path(path) is False

    def test_absolute_path_rejected(self):
        assert GitSource._is_safe_file_path("/etc/passwd") is False

    def test_null_byte_rejected(self):
        assert GitSource._is_safe_file_path("file\x00.py") is False

    @pytest.mark.asyncio
    async def test_get_file_content_rejects_traversal(self, tmp_path: Path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_repo(repo_dir)

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        with pytest.raises(GitSourceError, match="Unsafe file path"):
            await source.get_file_content("HEAD", "../../etc/passwd")


class TestSearchCode:
    """Tests for search_code method."""

    @pytest.mark.asyncio
    async def test_search_code_returns_matches(self, tmp_path: Path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_repo(repo_dir)

        # Add a file with searchable content
        code_file = repo_dir / "src" / "auth.py"
        code_file.parent.mkdir(parents=True)
        code_file.write_text(
            "def verify_token(token):\n    return check(token)\n", encoding="utf-8"
        )
        repo = Repo(repo_dir)
        repo.index.add(["src/auth.py"])
        repo.index.commit("add auth")

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        results = await source.search_code("verify_token", "HEAD")
        assert len(results) >= 1
        assert any(r["file"] == "src/auth.py" for r in results)
        assert any("verify_token" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_search_code_no_results(self, tmp_path: Path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_repo(repo_dir)

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        results = await source.search_code("nonexistent_pattern_xyz", "HEAD")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_code_with_dash_prefix_pattern(self, tmp_path: Path):
        """Pattern starting with '--' must not be interpreted as a git grep flag."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_repo(repo_dir)

        code_file = repo_dir / "config.py"
        code_file.write_text("value = '--exec=rm -rf /'\n", encoding="utf-8")
        repo = Repo(repo_dir)
        repo.index.add(["config.py"])
        repo.index.commit("add config")

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        # Should not raise or be misinterpreted as a flag
        results = await source.search_code("--exec", "HEAD")
        assert len(results) >= 1
        assert any("--exec" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_search_code_with_file_glob(self, tmp_path: Path):
        """file_glob parameter should filter search to matching files only."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_repo(repo_dir)

        py_file = repo_dir / "code.py"
        py_file.write_text("target_string = True\n", encoding="utf-8")
        txt_file = repo_dir / "data.txt"
        txt_file.write_text("target_string = False\n", encoding="utf-8")
        repo = Repo(repo_dir)
        repo.index.add(["code.py", "data.txt"])
        repo.index.commit("add files")

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        results = await source.search_code("target_string", "HEAD", file_glob="*.py")
        assert len(results) >= 1
        assert all(r["file"].endswith(".py") for r in results)

    @pytest.mark.asyncio
    async def test_search_code_raises_on_git_errors(self, tmp_path: Path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_repo(repo_dir)

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        source._repo = MagicMock()
        source._repo.commit.return_value = object()
        source._repo.git.grep.side_effect = GitCommandError(
            "git grep",
            2,
            stderr="fatal: grep failed",
        )

        with pytest.raises(GitSourceError, match="Failed to search code"):
            await source.search_code("README", "HEAD")

    @pytest.mark.asyncio
    async def test_search_code_uses_fixed_string_mode(self, tmp_path: Path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_repo(repo_dir)

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        source._repo = MagicMock()
        source._repo.commit.return_value = object()
        source._repo.git.grep.return_value = "HEAD:authlib/jose/jwt.py:10:Decode\\(dst)"

        results = await source.search_code(
            r"Decode\(dst",
            "HEAD",
            file_glob="authlib/jose/*.py",
        )
        assert results == [
            {
                "file": "authlib/jose/jwt.py",
                "line": 10,
                "content": r"Decode\(dst)",
            }
        ]
        source._repo.git.grep.assert_called_once()
        grep_args = source._repo.git.grep.call_args.args
        assert "-F" in grep_args


class TestListFiles:
    @pytest.mark.asyncio
    async def test_list_files_returns_paths(self, tmp_path: Path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_repo(repo_dir)

        nested = repo_dir / "src" / "api" / "auth.py"
        nested.parent.mkdir(parents=True)
        nested.write_text("print('ok')\n", encoding="utf-8")
        repo = Repo(repo_dir)
        repo.index.add(["src/api/auth.py"])
        repo.index.commit("add nested file")

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        files = await source.list_files("HEAD", max_files=20)
        assert "README.md" in files
        assert "src/api/auth.py" in files

    @pytest.mark.asyncio
    async def test_list_files_honors_max_depth(self, tmp_path: Path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_repo(repo_dir)

        nested = repo_dir / "src" / "api" / "auth.py"
        nested.parent.mkdir(parents=True)
        nested.write_text("print('ok')\n", encoding="utf-8")
        repo = Repo(repo_dir)
        repo.index.add(["src/api/auth.py"])
        repo.index.commit("add nested file")

        settings = get_settings(clone_base_dir=tmp_path / "clones")
        source = GitSource(str(repo_dir), settings)
        await source.ensure_cloned()

        files = await source.list_files("HEAD", max_files=20, max_depth=2)
        assert "README.md" in files
        assert "src/api/auth.py" not in files


@pytest.mark.asyncio
async def test_get_file_content_raises_on_unexpected_git_errors(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_repo(repo_dir)

    settings = get_settings(clone_base_dir=tmp_path / "clones")
    source = GitSource(str(repo_dir), settings)
    await source.ensure_cloned()

    source._repo = MagicMock()
    source._repo.git.show.side_effect = GitCommandError(
        "git show",
        128,
        stderr="fatal: bad object",
    )

    with pytest.raises(GitSourceError, match="Failed to read file"):
        await source.get_file_content("HEAD", "README.md")


class TestSlugCollisionResistance:
    """Verify that different URLs produce different local_path slugs."""

    def test_different_urls_different_slugs(self, tmp_path: Path):
        settings = get_settings(clone_base_dir=tmp_path)
        s1 = GitSource("https://example.com/repo.git", settings)
        s2 = GitSource("https://example.com/repo/git", settings)
        assert s1.local_path != s2.local_path

    def test_slug_contains_hash(self, tmp_path: Path):
        settings = get_settings(clone_base_dir=tmp_path)
        source = GitSource("https://example.com/repo", settings)
        # Slug should end with _<12-char-hex>
        slug = source.local_path.name
        parts = slug.rsplit("_", 1)
        assert len(parts) == 2
        assert len(parts[1]) == 12
