# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()  # pyright: ignore[reportPrivateImportUsage]
# Points to your src/ directory
src = Path(__file__).parent.parent / "src"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    # Handle __init__.py and __main__.py special cases
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Skip the root if it's empty
    if not parts:
        continue

    nav[parts] = doc_path.as_posix()

    # Create the virtual markdown file
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# {ident}\n\n::: {ident}")

    # Make the "edit" button on the docs page point to the actual Python
    # source file
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Generate the navigation summary
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
