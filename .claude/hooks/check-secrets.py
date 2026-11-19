#!/usr/bin/env python3
"""
PreToolUse hook for the Bash tool. Blocks Claude from:
  1. Running commands that contain a literal API key
  2. Staging .env files (excluding .env.example / .env.enterprise.example)
  3. Committing while a .env file is staged

Reads JSON on stdin like:
  {"tool_name":"Bash","tool_input":{"command":"git add .env"}}

Emits a JSON object on stdout to deny:
  {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                           "permissionDecision": "deny",
                           "permissionDecisionReason": "..."}}

Otherwise exits 0 with no output (= allow).
"""
import json
import os
import re
import subprocess
import sys


def deny(reason: str) -> None:
    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(out))
    sys.exit(0)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # malformed input -> allow (don't break Claude)

    cmd = (data.get("tool_input") or {}).get("command", "") or ""
    if not cmd:
        sys.exit(0)

    # 1. Literal API key patterns (length >= 20 to avoid false positives on
    #    bare prefixes that appear in error messages or examples).
    key_re = re.compile(
        r"(sk-ant-[A-Za-z0-9_-]{20,}"
        r"|sk-or-[A-Za-z0-9_-]{20,}"
        r"|sk-proj-[A-Za-z0-9_-]{20,}"
        r"|sk-[A-Za-z0-9]{20,}"
        r"|AIza[0-9A-Za-z_-]{35}"
        r"|xai-[A-Za-z0-9]{20,})"
    )
    if key_re.search(cmd):
        deny(
            "Blocked: command contains a literal API key pattern. "
            "Refusing to run to prevent leakage. Use environment variable "
            "references (e.g., $OPENAI_API_KEY) instead of inline keys."
        )

    # Identify git add / git stage / git commit
    git_add = re.search(r"\bgit\s+(add|stage)\b", cmd) is not None
    git_commit = re.search(r"\bgit\s+commit\b", cmd) is not None

    # 2a. Explicit .env path (any form like .env, .env.local) excluding templates
    if git_add:
        # Find every .env-ish token
        tokens = re.findall(r"(?:^|[\s/=])(\.env(?:\.[A-Za-z0-9_.-]+)?)", cmd)
        templates = {".env.example", ".env.enterprise.example"}
        suspect = [t for t in tokens if t not in templates]
        if suspect:
            deny(
                "Blocked: refusing to stage .env file(s): "
                f"{', '.join(suspect)}. These are gitignored to prevent "
                "leaking API keys. Use .env.example for templates."
            )

        # 2b. Wildcard staging while .env exists in the working tree
        if re.search(r"\bgit\s+add\s+(-[AuU]+|--all|--update|\.|\*)(\s|$)", cmd):
            cwd = os.getcwd()
            if os.path.isfile(os.path.join(cwd, ".env")):
                deny(
                    "Blocked: refusing 'git add .' / -A / -u while .env exists "
                    "in the working tree (would stage it). Stage specific "
                    "files explicitly instead."
                )

    # 3. git commit while .env is staged
    if git_commit:
        try:
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True, text=True, timeout=5,
            ).stdout.splitlines()
        except Exception:
            staged = []
        templates = {".env.example", ".env.enterprise.example"}
        offenders = [
            f for f in staged
            if re.fullmatch(r"\.env(?:\.[A-Za-z0-9_.-]+)?", f) and f not in templates
        ]
        if offenders:
            deny(
                "Blocked: .env file(s) currently staged: "
                f"{', '.join(offenders)}. Run 'git reset HEAD <file>' before "
                "committing."
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
