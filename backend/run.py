"""
Script để chạy server từ thư mục backend.
File này tự động setup Python path và import app để chạy uvicorn.
"""
import sys
from pathlib import Path

# Thêm thư mục cha vào Python path để backend được nhận là package
backend_dir = Path(__file__).parent
parent_dir = backend_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import app từ backend.main (sau khi đã setup path)
from backend.main import app

# Chạy uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

