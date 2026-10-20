"""Responsive Flask interface for the exported diabetes screening model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "best_diabetes_model.joblib"
METADATA_PATH = ARTIFACTS / "model_metadata.json"

app = Flask(__name__)

FEATURES = [
    ("HighBP", "Cao huyết áp", "binary"), ("HighChol", "Cholesterol cao", "binary"),
    ("CholCheck", "Đã kiểm tra cholesterol 5 năm gần đây", "binary"), ("BMI", "Chỉ số BMI", "bmi"),
    ("Smoker", "Đã hút ít nhất 100 điếu thuốc", "binary"), ("Stroke", "Từng bị đột quỵ", "binary"),
    ("HeartDiseaseorAttack", "Bệnh tim hoặc từng đau tim", "binary"), ("PhysActivity", "Có hoạt động thể chất", "binary"),
    ("Fruits", "Ăn trái cây mỗi ngày", "binary"), ("Veggies", "Ăn rau mỗi ngày", "binary"),
    ("HvyAlcoholConsump", "Uống rượu nhiều", "binary"), ("AnyHealthcare", "Có bảo hiểm/chăm sóc y tế", "binary"),
    ("NoDocbcCost", "Không gặp bác sĩ vì chi phí", "binary"), ("GenHlth", "Sức khỏe tổng quát (1 tốt – 5 kém)", "health"),
    ("MentHlth", "Số ngày sức khỏe tinh thần kém (0–30)", "days"), ("PhysHlth", "Số ngày sức khỏe thể chất kém (0–30)", "days"),
    ("DiffWalk", "Khó đi bộ/leo cầu thang", "binary"), ("Sex", "Giới tính", "sex"),
    ("Age", "Nhóm tuổi (1: 18–24 … 13: ≥80)", "age"), ("Education", "Học vấn (1–6)", "education"),
    ("Income", "Thu nhập (1–8)", "income"),
]


def load_assets():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy toàn bộ diabetes_model_comparison.ipynb trước.")
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def get_score(model, frame):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(frame)[0, 1])
    if hasattr(model, "decision_function"):
        value = float(model.decision_function(frame)[0])
        return 1 / (1 + np.exp(-np.clip(value, -30, 30)))
    return float(np.clip(model.predict(frame)[0], 0, 1))


@app.route("/", methods=["GET", "POST"])
def index():
    result, values, error = None, {}, None
    try:
        model, metadata = load_assets()
    except FileNotFoundError as exc:
        return render_template("index.html", features=FEATURES, result=None, values={}, error=str(exc), metadata=None)
    if request.method == "POST":
        try:
            values = {key: float(request.form[key]) for key, _, _ in FEATURES}
            ordered = metadata["feature_names"]
            frame = pd.DataFrame([[values[key] for key in ordered]], columns=ordered)
            probability = get_score(model, frame)
            level = "Nguy cơ cao" if probability >= 0.65 else "Nguy cơ trung bình" if probability >= 0.35 else "Nguy cơ thấp"
            result = {"probability": round(probability * 100, 1), "level": level,
                      "class_name": "high" if probability >= .65 else "medium" if probability >= .35 else "low"}
        except (KeyError, ValueError) as exc:
            error = f"Dữ liệu nhập chưa hợp lệ: {exc}"
    return render_template("index.html", features=FEATURES, result=result, values=values, error=error, metadata=metadata)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
