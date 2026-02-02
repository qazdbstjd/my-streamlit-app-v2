import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import time

# --- Setup ---
st.set_page_config(page_title="ObjectTrace - Created by Yunseong", layout="wide")

st.markdown("""
    <style>
    /* 폰트를 코딩 폰트 느낌으로 통일 */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"], .main {
        font-family: 'Roboto Mono', monospace !important;
        background-color: #ffffff; /* 흰색 배경 */
        color: #0000ff; /* 파란색 글자 */
    }

    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background-color: #0000ff !important; /* 파란색 배경 */
        border-right: 2px solid #0000ff;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important; /* 사이드바 내부 흰색 글자 */
    }

    /* 메인 타이틀 (강조) */
    .stHeading h1 {
        background-color: #0000ff; /* 배경 파랑 */
        color: #ffffff !important; /* 글자 흰색 */
        padding: 10px;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: -1px;
    }

    /* 버튼 스타일 (반전 효과) */
    .stButton>button {
        border: 2px solid #0000ff !important;
        background-color: #ffffff !important;
        color: #0000ff !important;
        border-radius: 0px !important;
        font-weight: bold;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        background-color: #0000ff !important;
        color: #ffffff !important;
    }

    /* 슬라이더 및 입력창 컬러 */
    .stSlider [data-baseweb="slider"] {
        background-color: #0000ff;
    }
    
    /* 구분선 스타일 */
    hr {
        border: none;
        border-top: 3px dashed #0000ff;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("📹 ObjectTrace - Created by Yunseong")
st.write("Upload a video to detect moving objects (Person, Car, etc.) using YOLOv8.")

# --- Sidebar ---
st.sidebar.header("Settings")
model_type = st.sidebar.selectbox("Select Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

# --- Main Logic ---

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

try:
    model = load_model(model_type)
    st.sidebar.success(f"Model {model_type} loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path) # Show original video

    if st.button("Start Analysis"):
        st.write("Analyzing...")
        
        cap = cv2.VideoCapture(video_path)
        
        # properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Output setup
        output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
        # Streamlit displays mp4 best with H264. OpenCV default might need conversion or specific codec.
        # We try mp4v first. If browser issues occur, we might need ffmpeg conversion.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        st_frame = st.empty() # Placeholder for real-time updates
        progress_bar = st.progress(0)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = model(frame, conf=confidence_threshold)
            annotated_frame = results[0].plot() # YOLO built-in plotting

            # Write to output video
            out.write(annotated_frame)

            # Display in Streamlit (convert BGR to RGB)
            # We display every 3rd frame to improve preview performance during processing, or all if feasible.
            # Showing every frame might slow down processing loop significantly.
            # Let's show it:
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            curr_frame += 1
            if frame_count > 0:
                progress_bar.progress(min(curr_frame / frame_count, 1.0))

        cap.release()
        out.release()
        
        st.success("Analysis Complete!")
        
        # Provide download button
        # Note: Browsers may not play 'mp4v' codec in <video> tag well, so we offer download primarily.
        # To display the result in st.video(), it usually needs h264 encoding. 
        # For now, we provide the file for download.
        
        with open(output_path, "rb") as file:
            btn = st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
