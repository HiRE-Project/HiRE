#!/usr/bin/env python3
"""Check source structure, dependency boundaries, configs, and documentation offline."""
import ast
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "hire": ROOT / "src/hire",
    "hire_dice_rl": ROOT / "backends/dice_rl/src/hire_dice_rl",
}
FORBIDDEN_CORE = {"hire_dice_rl", "gym", "gymnasium", "hydra", "omegaconf", "wandb"}


def main():
    errors = []
    sources = [p for folder in [*PACKAGES.values(), ROOT / "scripts", ROOT / "tests"]
               for p in folder.rglob("*.py")]
    for path in sources:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
            continue
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
            for module in modules:
                top, *parts = module.split(".")
                if path.is_relative_to(PACKAGES["hire"]) and top in FORBIDDEN_CORE:
                    errors.append(f"Core dependency violation: {path.relative_to(ROOT)} imports {module}")
                if top in PACKAGES:
                    candidate = PACKAGES[top].joinpath(*parts)
                    if not candidate.is_dir() and not candidate.with_suffix(".py").is_file():
                        errors.append(f"{path.relative_to(ROOT)}: missing local module {module}")
    for path in (ROOT / "configs").rglob("*.json"):
        json.loads(path.read_text())
    for path in (ROOT / "configs").rglob("*.xml"):
        ET.parse(path)
    documents = [ROOT / "README.md", *(ROOT / "docs").glob("*.md"), *(ROOT / "backends").glob("*/README.md")]
    for path in documents:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if "://" in link or link.startswith("mailto:"):
                continue
            target, _, anchor = link.partition("#")
            destination = path.parent / target if target else path
            if not destination.exists():
                errors.append(f"{path.relative_to(ROOT)}: broken link {link}")
            elif anchor and destination.suffix == ".md":
                headings = [re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
                            for heading in re.findall(r'^#+\s+(.+)$', destination.read_text(), re.M)]
                if anchor not in headings:
                    errors.append(f"{path.relative_to(ROOT)}: missing heading {link}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {len(sources)} Python files; package boundaries, configs, and documentation links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
