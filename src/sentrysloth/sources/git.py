"""Git source: clone, fetch, list tags, diff between refs."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError

from sentrysloth.config import Settings

logger = logging.getLogger(__name__)


class GitSourceError(Exception):
    pass


def extract_repo_name(repo_url: str) -> str:
    """Extract repository name from URL or local path."""
    clean = repo_url.rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]
    return clean.split("/")[-1]


class GitSource:
    """Manages git operations for a repository."""

    def __init__(self, repo_url: str, settings: Settings) -> None:
        self.repo_url = repo_url
        self.settings = settings
        self._repo: Repo | None = None

    @property
    def repo_name(self) -> str:
        """Extract repo name from URL or path."""
        return extract_repo_name(self.repo_url)

    @property
    def local_path(self) -> Path:
        """Local clone path for this repo."""
        # Combine sanitized name with a hash suffix to prevent collisions
        # (e.g. "repo.git" vs "repo/git" would produce the same slug without this)
        sanitized = re.sub(r"[^\w\-.]", "_", self.repo_url.rstrip("/"))
        hash_suffix = hashlib.sha256(self.repo_url.encode()).hexdigest()[:12]
        slug = f"{sanitized}_{hash_suffix}"
        return self.settings.clone_base_dir / slug

    @property
    def repo(self) -> Repo:
        if self._repo is None:
            raise GitSourceError("Repository not initialized. Call ensure_cloned() first.")
        return self._repo

    def is_local_path(self) -> bool:
        """Check if repo_url points to a local directory."""
        return Path(self.repo_url).is_dir()

    async def ensure_cloned(self) -> Path:
        """Clone or open the repository. Returns local path."""
        if self.is_local_path():
            try:
                self._repo = Repo(self.repo_url)
                logger.info("Opened local repo: %s", self.repo_url)
                return Path(self.repo_url)
            except InvalidGitRepositoryError as exc:
                raise GitSourceError(f"Not a git repository: {self.repo_url}") from exc

        local = self.local_path
        if local.exists() and (local / ".git").exists():
            try:
                self._repo = Repo(str(local))
                logger.info("Fetching updates for %s", self.repo_url)
                await asyncio.to_thread(self._repo.remotes.origin.fetch)
                return local
            except (InvalidGitRepositoryError, GitCommandError):
                logger.warning("Corrupt clone at %s, re-cloning", local)
                if not self._is_safe_clone_path(local):
                    raise GitSourceError(f"Refusing to delete unsafe clone path: {local}") from None
                shutil.rmtree(local)

        local.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning %s to %s", self.repo_url, local)
        try:
            self._repo = await asyncio.to_thread(
                Repo.clone_from,
                self.repo_url,
                str(local),
            )
        except GitCommandError as exc:
            raise GitSourceError(f"Failed to clone {self.repo_url}: {exc}") from exc
        return local

    def _is_safe_clone_path(self, path: Path) -> bool:
        base = self.settings.clone_base_dir.resolve()
        candidate = path.resolve()
        return candidate != base and candidate.is_relative_to(base)

    async def list_tags(self, limit: int = 20) -> list[str]:
        """List tags sorted by commit date (newest first).

        Some repositories contain tags that point to non-commit objects (blob/tree).
        Those tags are silently skipped.
        """
        tags_with_dates = await self.list_tags_with_dates(limit=limit)
        return [name for name, _dt in tags_with_dates]

    async def list_tags_with_dates(self, limit: int = 20) -> list[tuple[str, datetime]]:
        """List recent tags with their commit dates.

        Returns: [(tag_name, committed_datetime), ...] sorted newest-first.
        """

        def _inner() -> list[tuple[str, datetime]]:
            tags: list[tuple[str, datetime]] = []
            for tag_ref in self.repo.tags:
                try:
                    dt = tag_ref.commit.committed_datetime
                except ValueError:
                    # Non-commit tags exist in the wild (e.g. refs to blobs/trees).
                    # Skip so batch/list commands never crash, but keep traceability.
                    logger.debug("Skipping non-commit tag: %s", tag_ref.name)
                    continue
                tags.append((tag_ref.name, dt))

            tags.sort(key=lambda x: x[1], reverse=True)
            return tags[:limit]

        return await asyncio.to_thread(_inner)

    async def get_diff(self, from_ref: str, to_ref: str) -> str:
        """Get unified diff between two refs."""
        await self._validate_ref(from_ref)
        await self._validate_ref(to_ref)
        try:
            return await asyncio.to_thread(self.repo.git.diff, from_ref, to_ref, unified=5)
        except GitCommandError as exc:
            raise GitSourceError(f"Failed to diff {from_ref}..{to_ref}: {exc}") from exc

    async def get_diff_stat(self, from_ref: str, to_ref: str) -> dict[str, int]:
        """Get diff statistics: files changed, additions, deletions."""
        await self._validate_ref(from_ref)
        await self._validate_ref(to_ref)
        try:
            numstat = await asyncio.to_thread(self.repo.git.diff, from_ref, to_ref, numstat=True)
        except GitCommandError as exc:
            raise GitSourceError(f"Failed to get diff stat: {exc}") from exc

        result = {"files_changed": 0, "additions": 0, "deletions": 0}
        for line in numstat.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            added, deleted, _file_path = parts[0], parts[1], parts[2]
            # Binary files show "-" for additions/deletions
            result["additions"] += int(added) if added != "-" else 0
            result["deletions"] += int(deleted) if deleted != "-" else 0
            result["files_changed"] += 1
        return result

    async def get_file_content(self, ref: str, file_path: str) -> str | None:
        """Get file content at a specific ref."""
        await self._validate_ref(ref)
        if not self._is_safe_file_path(file_path):
            raise GitSourceError(f"Unsafe file path rejected: {file_path}")
        try:
            return await asyncio.to_thread(self.repo.git.show, f"{ref}:{file_path}")
        except GitCommandError as exc:
            # Missing files should return None for callers that probe optional context.
            if self._is_missing_file_error(exc):
                return None
            raise GitSourceError(f"Failed to read file {file_path} at {ref}: {exc}") from exc

    @staticmethod
    def _is_safe_file_path(file_path: str) -> bool:
        """Reject path traversal attempts: .., absolute paths, null bytes."""
        if "\x00" in file_path:
            return False
        if file_path.startswith("/"):
            return False
        # Reject any component that is ".."
        parts = file_path.replace("\\", "/").split("/")
        return ".." not in parts

    async def search_code(
        self,
        pattern: str,
        ref: str,
        *,
        file_glob: str = "",
        max_results: int = 20,
    ) -> list[dict]:
        """Search code in repo at a given ref using git grep.

        Returns list of {"file": str, "line": int, "content": str}.
        """
        await self._validate_ref(ref)
        args = ["-n", "--max-count", str(max_results), "-F", "-e", pattern, ref]
        if file_glob:
            args.extend(["--", file_glob])
        try:
            raw = await asyncio.to_thread(self.repo.git.grep, *args)
        except GitCommandError as exc:
            # git grep returns exit code 1 when no matches found
            if getattr(exc, "status", None) == 1:
                return []
            raise GitSourceError(f"Failed to search code at {ref}: {exc}") from exc

        results: list[dict] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            # Format: ref:file_path:line_no:content
            # Split on ":" with max 4 parts (content may contain colons)
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue
            _ref, file_path, line_no, content = parts
            try:
                results.append(
                    {
                        "file": file_path,
                        "line": int(line_no),
                        "content": content,
                    }
                )
            except ValueError:
                continue
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _is_missing_file_error(exc: GitCommandError) -> bool:
        text = f"{exc}".lower()
        missing_markers = (
            "does not exist in",
            "exists on disk, but not in",
            "path '",
        )
        return any(marker in text for marker in missing_markers)

    async def list_files(
        self,
        ref: str,
        *,
        max_files: int = 1000,
        max_depth: int | None = None,
    ) -> list[str]:
        """List repository file paths at a ref.

        Uses ``git ls-tree`` and returns paths relative to repo root.
        """
        await self._validate_ref(ref)
        try:
            raw = await asyncio.to_thread(
                self.repo.git.ls_tree,
                "-r",
                "--name-only",
                ref,
            )
        except GitCommandError as exc:
            raise GitSourceError(f"Failed to list files at {ref}: {exc}") from exc

        results: list[str] = []
        for line in raw.splitlines():
            path = line.strip()
            if not path:
                continue
            if max_depth is not None and path.count("/") + 1 > max_depth:
                continue
            results.append(path)
            if len(results) >= max_files:
                break
        return results

    async def _validate_ref(self, ref: str) -> None:
        """Validate that a ref exists in the repo."""
        try:
            await asyncio.to_thread(self.repo.commit, ref)
        except (GitCommandError, ValueError, BadName) as exc:
            raise GitSourceError(f"Invalid ref: {ref}") from exc
