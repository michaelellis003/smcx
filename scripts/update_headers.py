# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Ensure all Python source files carry SPDX license headers."""

import argparse
import os
import subprocess
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_types = [("*.py", "# {}")]


def git_filenames(pattern: str) -> list[str]:
    """Return tracked and nonignored untracked files matching a pattern."""
    output = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            pattern,
        ],
        cwd=root,
    )
    return [
        os.path.join(root, os.fsdecode(filename))
        for filename in sorted(output.split(b"\0"))
        if filename
    ]


parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
dirty = []

for basename, comment in file_types:
    copyright_line = comment.format(
        "Copyright Contributors to the smcx project.\n"
    )
    # See https://spdx.org/ids-how
    spdx_line = comment.format("SPDX-License-Identifier: Apache-2.0\n")

    filenames = git_filenames(basename)
    for filename in filenames:
        # A file may vanish after Git reports the worktree snapshot.
        try:
            with open(filename) as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue

        # Ignore empty files like __init__.py
        if all(line.isspace() for line in lines):
            continue

        # Ensure first few lines are copyright notices.
        changed = False
        lineno = 0
        if not lines[lineno].startswith(comment.format("Copyright")):
            lines.insert(lineno, copyright_line)
            changed = True
        lineno += 1
        # A file may end inside the copyright block (for example a
        # single copyright line with no SPDX identifier yet).
        while lineno < len(lines) and lines[lineno].startswith(
            comment.format("Copyright")
        ):
            lineno += 1

        # Ensure next line is an SPDX short identifier.
        if lineno >= len(lines) or not lines[lineno].startswith(
            comment.format("SPDX-License-Identifier")
        ):
            lines.insert(lineno, spdx_line)
            changed = True
        lineno += 1

        # Ensure next line is blank.
        if lineno < len(lines) and not lines[lineno].isspace():
            lines.insert(lineno, "\n")
            changed = True

        if not changed:
            continue

        if args.check:
            dirty.append(filename)
            continue

        # Omit O_CREAT so a file removed since reading is not recreated.
        try:
            descriptor = os.open(filename, os.O_WRONLY | os.O_TRUNC)
        except FileNotFoundError:
            continue
        with os.fdopen(descriptor, "w") as f:
            f.write("".join(lines))

        print(f"updated {filename[len(root) + 1 :]}")

if dirty:
    missing = "\n".join(dirty)
    print(f"The following files need license headers:\n{missing}")
    print("Please run 'make license'")
    sys.exit(1)
