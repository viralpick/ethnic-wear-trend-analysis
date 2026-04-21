"""파일 IO 유틸. 현재는 atomic JSON write 만 담고 있다."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, content: str) -> None:
    """같은 디렉토리에 temp 파일로 쓰고 rename. 파셜 파일이 읽는 쪽에 노출되는 걸 방지한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_json_atomic(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))
