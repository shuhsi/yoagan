import streamlit as st
from ultralytics import YOLO  # 假设用 YOLO 模型
import cv2
import numpy as np

# 1. 页面标题
st.title("屋顶分割模型演示")

# 2. 加载模型（注意模型在 GitHub 中的相对路径）
@st.cache_resource  # 缓存模型，避免重复加载
def load_model():
    model = YOLO("building_best_seg_t4.pt")  # 模型文件名需与 GitHub 中一致
    return model

model = load_model()

# 3. 上传图片功能
uploaded_file = st.file_uploader("上传图片", type=["jpg", "png"])

# 4. 预测与显示结果
if uploaded_file is not None:
    # 读取图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转 RGB 格式
    
    # 模型预测
    results = model(img)
    
    # 显示结果
    st.subheader("预测结果")
    st.image(results[0].plot(), caption="分割效果", use_column_width=True)
