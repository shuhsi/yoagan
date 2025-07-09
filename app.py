import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

# 加载模型（与app.py同目录）
model = YOLO("building_best_seg_t4.pt")

st.title("屋顶分割工具")
uploaded_file = st.file_uploader("选择图像", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    st.subheader("原始图像")
    st.image(image, use_column_width=True)

    with st.spinner("分割中..."):
        results = model(img_np, imgsz=640, conf=0.3)
        res_plotted = results[0].plot(masks=True)
        res_image = Image.fromarray(res_plotted[..., ::-1])

    st.subheader("分割结果")
    st.image(res_image, use_column_width=True)

    # 保存按钮
    buf = io.BytesIO()
    res_image.save(buf, format="PNG")
    st.download_button(
        "保存结果",
        buf.getvalue(),
        "roof_result.png",
        "image/png"
    )