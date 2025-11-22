"""
Cho phép chạy backend như một module: python -m backend
"""
import sys
from pathlib import Path

# Thêm thư mục cha vào Python path
backend_dir = Path(__file__).parent
parent_dir = backend_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import và chạy
from backend.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

