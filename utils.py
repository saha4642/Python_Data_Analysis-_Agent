from __future__ import annotations

import re
from typing import Optional


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> Optional[str]:
    """
    Extract the first python code block from the model output, if any.
    """
    m = CODE_BLOCK_RE.search(text or "")
    if not m:
        return None
    return m.group(1).strip() or None
