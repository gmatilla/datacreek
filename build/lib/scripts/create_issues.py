#!/usr/bin/env python3
"""Automate GitHub issue creation from AGENTS backlog."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import requests  # type: ignore[import-untyped]

BACKLOG_FILE = Path(__file__).resolve().parents[1] / "AGENTS.md"

TASK_RE = re.compile(r"^\* \[ \] (.+)")
SECTION_RE = re.compile(r"^##+ (.+)")


def parse_tasks(text: Iterable[str]) -> list[tuple[str, str]]:
    """Return (section, task) tuples for unchecked tasks."""
    section = ""
    tasks: list[tuple[str, str]] = []
    for line in text:
        header = SECTION_RE.match(line)
        if header:
            section = header.group(1)
            continue
        m = TASK_RE.match(line)
        if m:
            tasks.append((section, m.group(1).strip()))
    return tasks


def create_issue(repo: str, token: str, section: str, title: str) -> None:
    """Create a GitHub issue with hardening labels."""
    url = f"https://api.github.com/repos/{repo}/issues"
    payload = {
        "title": title,
        "labels": ["hardening", f"area/{section.lower().split()[0]}"],
    }
    headers = {"Authorization": f"token {token}"}
    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    if resp.status_code >= 300:
        raise SystemExit(f"GitHub API error: {resp.text}")
    issue = resp.json()
    print(f"Created issue #{issue['number']}: {issue['html_url']}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="owner/repo slug")
    args = parser.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN environment variable is required")
    tasks = parse_tasks(BACKLOG_FILE.read_text().splitlines())
    for section, title in tasks:
        create_issue(args.repo, token, section, title)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
