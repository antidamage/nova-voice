from __future__ import annotations

from pathlib import Path


def load_skills(root: Path) -> str:
    if not root.exists():
        return ""
    sections: list[str] = []
    for path in sorted(root.glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                text = parts[2]
        sections.append(text.strip())
    return "\n\n".join(sections)
