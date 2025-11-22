================================================================================
                    BEAUTY EDITOR PRO - HƯỚNG DẪN SỬ DỤNG
================================================================================

MÔ TẢ DỰ ÁN
-----------
Beauty Editor Pro là ứng dụng chỉnh sửa ảnh làm đẹp khuôn mặt sử dụng AI.
Ứng dụng bao gồm:
- Frontend: React + TypeScript + Vite
- Backend: FastAPI + Python với MediaPipe và OpenCV

TÍNH NĂNG
---------
- Phân tích khuôn mặt tự động
- Làm mịn da (Skin Smoothing)
- Làm sáng da (Skin Brightening)
- Chỉnh sửa mắt (Eye Enhancement)
- Chỉnh sửa môi (Lip Enhancement)
- Chỉnh sửa mũi (Nose Reshaping)
- Chỉnh sửa khuôn mặt (Face Reshaping)
- Chỉnh sửa cằm (Chin Reshaping)

YÊU CẦU HỆ THỐNG
----------------
Frontend:
- Node.js >= 18.0.0
- npm hoặc yarn

Backend:
- Python >= 3.9
- pip

CÀI ĐẶT
--------

1. CÀI ĐẶT FRONTEND:
   -----------------
   a. Cài đặt dependencies:
      npm install
   
   b. Tạo file .env.local (nếu cần):
      GEMINI_API_KEY=your_gemini_api_key_here
   
   c. Chạy development server:
      npm run dev
   
   Frontend sẽ chạy tại: http://localhost:5173

2. CÀI ĐẶT BACKEND:
   -----------------
   a. Tạo virtual environment (khuyến nghị):
      python -m venv venv
   
   b. Kích hoạt virtual environment:
      Windows: venv\Scripts\activate
      Linux/Mac: source venv/bin/activate
   
   c. Cài đặt dependencies:
      pip install -r requirements.txt
   
   d. Chạy backend server:
      Từ thư mục gốc:
      uvicorn backend.main:app --reload --port 8000
      
      Hoặc từ thư mục backend:
      cd backend
      uvicorn main:app --reload --port 8000
   
   Backend sẽ chạy tại: http://localhost:8000
   API Documentation: http://localhost:8000/docs

CẤU TRÚC DỰ ÁN
---------------
beauty-editor-pro2/
├── backend/                 # Backend Python
│   ├── __init__.py
│   ├── main.py             # FastAPI application
│   ├── config.py           # Settings và CORS config
│   ├── models.py           # Pydantic models
│   └── beauty_pipeline.py  # Xử lý làm đẹp
├── components/             # React components
│   ├── Canvas.tsx
│   ├── Header.tsx
│   ├── LeftSidebar.tsx
│   ├── Sidebar.tsx
│   └── SliderControl.tsx
├── utils/                  # Utilities
│   ├── aiProClient.ts
│   ├── geminiImage.ts
│   └── imageProcessor.ts
├── App.tsx                 # Main App component
├── index.tsx               # Entry point
├── package.json            # Frontend dependencies
├── requirements.txt        # Backend dependencies
└── vite.config.ts         # Vite configuration

API ENDPOINTS
-------------
Backend cung cấp các API sau:

1. GET /health
   - Kiểm tra trạng thái server
   - Response: {"status": "ok"}

2. POST /api/beauty/analyze
   - Phân tích khuôn mặt trong ảnh
   - Input: image (file upload)
   - Output: FaceAnalysisResponse với thông tin khuôn mặt

3. POST /api/beauty/apply
   - Áp dụng các hiệu ứng làm đẹp
   - Input: 
     * image (file upload)
     * beautyConfig (JSON string)
   - Output: BeautyResponse với ảnh đã xử lý (base64)

CẤU HÌNH CORS
-------------
Backend đã cấu hình CORS cho các origin sau:
- http://localhost:3000
- http://localhost:5173
- http://127.0.0.1:3000
- http://127.0.0.1:5173

Nếu cần thêm origin khác, chỉnh sửa file backend/config.py

SỬ DỤNG
-------
1. Khởi động backend trước:
   
   Từ thư mục gốc (khuyến nghị):
   uvicorn backend.main:app --reload --port 8000
   
   Hoặc từ thư mục backend:
   cd backend
   uvicorn main:app --reload --port 8000

2. Khởi động frontend:
   npm run dev

3. Cấu hình API URL (tùy chọn):
   Tạo file .env.local trong thư mục gốc:
   VITE_API_BASE_URL=http://localhost:8000
   
   Nếu không có file .env.local, frontend sẽ mặc định sử dụng:
   http://localhost:8000

4. Mở trình duyệt và truy cập:
   http://localhost:3000 (hoặc port mà Vite hiển thị)

5. Upload ảnh và sử dụng các công cụ làm đẹp
   
   Lưu ý: Frontend sẽ tự động kiểm tra backend có sẵn sàng không.
   Nếu backend không khả dụng, frontend sẽ fallback về xử lý client-side.

GHI CHÚ
-------
- Đảm bảo backend đang chạy trước khi sử dụng frontend
- Ảnh phải có khuôn mặt rõ ràng để ứng dụng hoạt động tốt
- Các tham số chỉnh sửa có thể điều chỉnh bằng slider trong giao diện
- API documentation có sẵn tại http://localhost:8000/docs khi backend đang chạy

HỖ TRỢ
-------
Nếu gặp vấn đề:
1. Kiểm tra console của trình duyệt (F12)
2. Kiểm tra logs của backend server
3. Đảm bảo tất cả dependencies đã được cài đặt đúng
4. Kiểm tra Python version (>= 3.9) và Node.js version (>= 18)

================================================================================
                              CHÚC BẠN SỬ DỤNG VUI VẺ!
================================================================================

