"""Generate mkdocstrings reference pages for swarm-lab packages."""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

PACKAGE_ROOTS = {
    "environments": Path("environments/environments"),
    "scripts": Path("scripts"),
}
"""Import package names mapped to their source roots."""


nav = mkdocs_gen_files.Nav()

for package_name, source_root in PACKAGE_ROOTS.items():
    for path in sorted(source_root.rglob("*.py")):
        module_path = path.relative_to(source_root).with_suffix("")
        parts = (package_name, *module_path.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = Path("reference", *parts, "index.md")
        else:
            doc_path = Path("reference", *parts).with_suffix(".md")

        identifier = ".".join(parts)
        nav[parts] = doc_path.relative_to("reference").as_posix()

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            fd.write(f"# `{identifier}`\n\n")
            fd.write(f"::: {identifier}\n")
            fd.write("    options:\n")
            fd.write("      show_source: true\n")

        mkdocs_gen_files.set_edit_path(doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
