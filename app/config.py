import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / 'config.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
