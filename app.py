import os

# Tắt cảnh báo bảo mật
os.environ["TRUST_REMOTE_CODE"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from anomalib.deploy import TorchInferencer

# GIAO DIỆN

st.set_page_config(page_title="AI QA/QC Inspector", page_icon="🏭", layout="wide")

# Làm đẹp giao diện bằng CSS
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: 800; color: #1E3A8A; text-align: center; margin-bottom: 5px;}
    .sub-title { font-size: 18px; color: #6B7280; text-align: center; margin-bottom: 30px;}
    .stButton>button { border-radius: 8px; font-weight: bold; height: 50px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏭 Hệ thống nhận diện sản phẩm lỗi trong công nghiệp (AI QA/QC Inspector)</div>', unsafe_allow_html=True)
st.markdown("---")

# THANH ĐIỀU KHIỂN

with st.sidebar:
    st.header("Bảng Điều Khiển")
    st.write("Cấu hình đầu vào cho hệ thống.")
    
    model_dir = r"E:\dai hoc\project\exported_models"
    
    try:
        product_list = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    except Exception:
        product_list = ["Lỗi đường dẫn thư mục!"]
        
    selected_product = st.selectbox("Chọn loại sản phẩm:", product_list)
    MODEL_PATH = rf"E:\dai hoc\project\exported_models\{selected_product}\weights\torch\model.pt"

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Khu vực tải ảnh
    uploaded_file = st.file_uploader("Tải ảnh sản phẩm", type=["png", "jpg", "jpeg"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    check_button = st.button("PHÂN TÍCH NGAY", use_container_width=True, type="primary")

# NẠP MÔ HÌNH VÀO BỘ NHỚ

@st.cache_resource
def load_model(path):
    return TorchInferencer(path=path, device="cuda")

try:
    inferencer = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Không thể tải mô hình cho '{selected_product}'. Vui lòng kiểm tra lại. Lỗi: {e}")
    st.stop()

# 4. KHU VỰC HIỂN THỊ KẾT QUẢ

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    original_img_array = np.array(image)

    if check_button:
        with st.spinner("đang quét bề mặt sản phẩm..."):
            
            # CHẠY DỰ ĐOÁN
            predictions = inferencer.predict(image=original_img_array)
            
            # VẼ HEATMAP
            anomaly_map = predictions.anomaly_map
            if hasattr(anomaly_map, "detach"):
                anomaly_map = anomaly_map.detach().cpu().numpy()
            else:
                anomaly_map = np.array(anomaly_map)
                
            anomaly_map = np.squeeze(anomaly_map)
            map_normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(map_normalized, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap_resized = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
            overlay_img = cv2.addWeighted(original_img_array, 0.5, heatmap_resized, 0.5, 0)

            # LẤY ĐIỂM SỐ
            score = predictions.pred_score.item() if hasattr(predictions.pred_score, 'item') else float(predictions.pred_score)

            # HIỂN THỊ TRẠNG THÁI
            st.subheader("Kết Quả Đánh Giá")
            
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                if predictions.pred_label:
                    st.error("SẢN PHẨM CÓ LỖI")
                else:
                    st.success("SẢN PHẨM ĐẠT CHUẨN")
                    
            with res_col2:
                st.metric(
                    label="Độ lệch cấu trúc", 
                    value=f"{score:.4f}", 
                    delta="Vượt ngưỡng an toàn" if predictions.pred_label else "Trong ngưỡng an toàn",
                    delta_color="inverse"
                )

            st.markdown("---")
            
            # HIỂN THỊ HAI ẢNH SO SÁNH
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(image, caption="Ảnh Gốc Thu Ghi Từ Camera", use_container_width=True)
            with img_col2:
                st.image(overlay_img, caption="Vùng Lỗi Được Định Vị", use_container_width=True)
else:
    # Màn hình chờ khi chưa tải ảnh
    st.info("Vui lòng chọn loại sản phẩm và tải ảnh lên từ Bảng điều khiển bên trái để bắt đầu.")