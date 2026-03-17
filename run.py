import json
from pathlib import Path
import uvicorn

config = json.loads(Path('config.json').read_text(encoding='utf-8'))

if __name__ == '__main__':
    uvicorn.run('app.main:app', host=config['host'], port=config['port'], reload=True)
