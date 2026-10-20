# Sàng lọc nguy cơ tiểu đường

Dự án so sánh 9 phương pháp trên dữ liệu BRFSS 2015, sau đó tự chọn mô hình có ROC-AUC cao nhất để triển khai giao diện Flask responsive.

## Chạy dự án

```powershell
python -m pip install -r requirements.txt
# Mở diabetes_model_comparison.ipynb và Run All để tạo model
python app.py
```

Mở `http://127.0.0.1:5000` trên desktop hoặc điện thoại cùng mạng (dùng địa chỉ IPv4 của máy tính thay cho `127.0.0.1`).

- Tập train: `data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv`
- Tập test: `data/diabetes_binary_health_indicators_BRFSS2015.csv`
- Kết quả sau khi chạy notebook: `artifacts/best_diabetes_model.joblib`, bảng chỉ số CSV, ROC, biểu đồ so sánh và ma trận nhầm lẫn.

`diabetes_model_comparison.ipynb` là notebook duy nhất chứa toàn bộ mã đọc dữ liệu, train, đánh giá, vẽ biểu đồ và lưu model. PCA là kỹ thuật giảm chiều nên được đánh giá trong pipeline **PCA + Logistic Regression**; K-Means được ánh xạ cụm sang nguy cơ theo tỷ lệ nhãn trong train. Hồi quy tuyến tính được dùng tham khảo, với đầu ra được chặn về thang điểm nguy cơ 0–1.

Mô hình triển khai được chọn bằng ROC-AUC trên validation hold-out (20% tách từ file train), rồi fit lại trên toàn bộ train. Điều này cần thiết vì file BRFSS đầy đủ có thể chứa các bản ghi đã được lấy vào bộ train 50/50; các chỉ số trên file đầy đủ được lưu riêng để tham khảo, không dùng để chọn model.

## Deploy trên Render

Repository đã có sẵn `render.yaml`. Trên Render chọn **New → Blueprint**, kết nối repository này và deploy. Nếu tạo Web Service thủ công, dùng:

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app`
- Health Check Path: `/health`

Model được nén và commit cùng repository để Render không phải huấn luyện lại khi build.

Kết quả chỉ phục vụ học tập và hỗ trợ sàng lọc, không phải chẩn đoán y khoa.
