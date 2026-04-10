# Copyright Contributors to the smcjax project.
# SPDX-License-Identifier: Apache-2.0

"""Ensure all Python source files carry SPDX license headers."""

import argparse
import glob
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
blacklist = ["/build/", "/dist/", "/smcjax.egg", "/venv/", "/.venv/"]
file_types = [("*.py", "# {}")]

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
dirty = []

for basename, comment in file_types:
    copyright_line = comment.format(
        "Copyright Contributors to the smcjax project.\n"
    )
    # See https://spdx.org/ids-how
    spdx_line = comment.format("SPDX-License-Identifier: Apache-2.0\n")

    filenames = glob.glob(os.path.join(root, "**", basename), recursive=True)
    filenames.sort()
    filenames = [
        filename
        for filename in filenames
        if not any(word in filename for word in blacklist)
    ]
    for filename in filenames:
        with open(filename) as f:
            lines = f.readlines()

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
        while lines[lineno].startswith(comment.format("Copyright")):
            lineno += 1

        # Ensure next line is an SPDX short identifier.
        if not lines[lineno].startswith(
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

        with open(filename, "w") as f:
            f.write("".join(lines))

        print(f"updated {filename[len(root) + 1 :]}")

if dirty:
    missing = "\n".join(dirty)
    print(f"The following files need license headers:\n{missing}")
    print("Please run 'make license'")
    sys.exit(1)
