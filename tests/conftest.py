"""共通 fixtures。"""

import sys
from pathlib import Path

# repo root を sys.path に追加 (src/ を import するため)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
