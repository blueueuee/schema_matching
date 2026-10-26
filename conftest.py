import sys
from pathlib import Path

# Ensure repository root is on sys.path for test discovery and package imports
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
