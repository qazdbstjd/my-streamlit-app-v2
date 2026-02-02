import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import time
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="Created by Yun Seong #1 : 📹OBJECT TRACE", layout="wide")

# 2. 강력한 레트로 브루탈리즘 CSS 적용
st.markdown("""
    <style>
    /* 1. Source Code Pro 폰트 불러오기 */
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;700;900&display=swap');

    /* 2. 전체 요소에 적용 */
    html, body, [class*="css"], .main, stMarkdown, h1, p, button {
        font-family: 'Source Code Pro', monospace !important;
    }

    /* 사이드바 스타일링 */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 3px solid #00ffff !important;
    }

    /* 모든 텍스트 컬러를 사이언(Cyan)으로 고정 */
    h1, h2, h3, h4, p, label, .stMarkdown, span, [data-testid="stMetricLabel"] {
        color: #00ffff !important;
        text-transform: uppercase !important;
    }

    /* 메인 타이틀 박스 스타일 */
    .title-container {
        border: 4px solid #00ffff;
        padding: 20px;
        margin-bottom: 30px;
        display: inline-block;
        background-color: #000000;
    }
    .title-main {
        font-size: 40px !important;
        font-weight: 900 !important;
        line-height: 1.2;
        margin: 0;
        color: #00ffff !important;
    }

    /* 버튼 스타일 (반전 효과) */
    div.stButton > button:first-child {
        width: 100%;
        background-color: #000000 !important;
        color: #00ffff !important;
        border: 3px solid #00ffff !important;
        border-radius: 0px !important;
        font-weight: 800 !important;
        padding: 15px !important;
        transition: 0.2s;
        text-transform: uppercase;
    }
    div.stButton > button:first-child:hover {
        background-color: #00ffff !important;
        color: #000000 !important;
    }

    /* 메트릭 박스 커스텀 */
    [data-testid="stMetric"] {
        border: 2px solid #00ffff;
        padding: 15px;
        background-color: #000000;
    }
    [data-testid="stMetricValue"] {
        color: #00ffff !important;
    }

    /* 업로더 및 슬라이더 스타일 */
    .stFileUploader, .stSlider {
        border: 1px dashed #00ffff;
        padding: 10px;
    }
    
    /* 구분선 */
    hr {
        border-top: 4px double #00ffff !important;
    }

    /* Success/Error override */
    .stAlert {
        background-color: #000000 !important;
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
    }

    /* 4. 마우스 커서 십자선 및 반전 설정 */
    html, body, .main {
        cursor: crosshair !important;
    }

    /* 클릭 요소 위에 있을 때 커서 반전 효과 */
    button, a, [data-testid="stFileUploadDropzone"], .stSlider {
        mix-blend-mode: difference; 
        cursor: crosshair !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 모델 로드 (캐시 사용)
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# 4. 사이드바 제어
with st.sidebar:
    st.markdown("### 01. SETUP")
    model_type = st.selectbox("MODEL_SELECT", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    confidence_threshold = st.slider("THRESHOLD", 0.0, 1.0, 0.4, 0.05)
    st.write("---")
    st.markdown("### STATUS: ACTIVE")

    try:
        model = load_model(model_type)
        st.success(f"SUCCESS: {model_type} LOADED")
    except Exception as e:
        st.error(f"ERROR: {e}")

# 5. 메인 화면 구성
st.markdown('<div class="title-container"><p class="title-main">CREATED BY YUN SEONG #1 : 📹 OBJECT TRACE</p></div>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown("#### // VIDEO_INPUT")
    uploaded_file = st.file_uploader("UPLOAD_FILE", type=["mp4", "mov", "avi"])
    video_placeholder = st.empty()

with col_right:
    st.markdown("#### // ANALYTICS")
    metric_placeholder = st.empty()
    metric_placeholder.metric(label="ENTITIES_DETECTED", value="00", delta="SCANNING")
    st.write("---")
    status_text = st.empty()
    status_text.info("SYSTEM_READY: WAITING FOR INPUT...")

# 6. 분석 엔진 실행
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    with col_left:
        video_placeholder.video(video_path)

    with col_right:
        if st.button("START ANALYSIS"):
            status_text.warning("ANALYZING...")
            
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            st_frame = st.empty() 
            progress_bar = st.progress(0)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            curr_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=confidence_threshold)
                annotated_frame = results[0].plot() 
                out.write(annotated_frame)

                with col_left:
                    st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                with col_right:
                    obj_count = len(results[0].boxes)
                    metric_placeholder.metric("ENTITIES_DETECTED", f"{obj_count:02d}", delta="ACTIVE")
                
                curr_frame += 1
                if frame_count > 0:
                    progress_bar.progress(min(curr_frame / frame_count, 1.0))

            cap.release()
            out.release()
            
            status_text.success("SUCCESS: ANALYSIS_COMPLETE")
            
            with open(output_path, "rb") as file:
                st.download_button(
                    label="DOWNLOAD_PROCESSED_VIDEO",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
