import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict

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
        border: 2px solid #00ffff;
        padding: 20px;
        margin-bottom: 30px;
        display: inline-block;
        background-color: #000000;
    }
    .title-main {
        font-size: 40px !important;
        font-weight: 500 !important;
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

# 6. 분석 엔진 실행 (궤적 추적 기능 통합)
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

            # 궤적 저장을 위한 딕셔너리 (ID별 좌표 리스트)
            track_history = defaultdict(lambda: [])

            while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- 이 부분을 추가하세요 ---
            # 영상을 640px 너비로 리사이징하여 연산량 감소
            analysis_frame = cv2.resize(frame, (640, int(height * (640 / width))))
            # --------------------------

            # 아래 model.track의 대상을 frame에서 analysis_frame으로 변경
            results = model.track(analysis_frame, persist=True, conf=confidence_threshold)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 객체 추적 실행 (persist=True 필수)
                results = model.track(frame, persist=True, conf=confidence_threshold)
                
                # YOLO 기본 바운딩 박스 그리기
                annotated_frame = results[0].plot() 

                # 궤적 시각화 로직 시작
                if results[0].boxes.id is not None:
                    # 박스 정보(xywh)와 할당된 ID 추출
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y))) # 중심점 추가
                        
                        # 궤적 길이 조절 (최근 30프레임 흔적 유지)
                        if len(track) > 20:
                            track.pop(0)

                        # 이동 경로 선(Line) 그리기
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 0), thickness=2)
                        
                        # 현재 위치 점(Dot) 찍기
                        cv2.circle(annotated_frame, (int(x), int(y)), 4, (0, 0, 255), -1)

                # 분석 프레임 비디오 파일로 저장
                out.write(annotated_frame)

                # 메인 화면 실시간 렌더링
                with col_left:
                    st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                # 우측 지표 실시간 업데이트
                with col_right:
                    obj_count = len(results[0].boxes) if results[0].boxes.id is not None else 0
                    metric_placeholder.metric("ENTITIES_DETECTED", f"{obj_count:02d}", delta="ACTIVE")
                
                # 진행률 표시
                curr_frame += 1
                if frame_count > 0:
                    progress_bar.progress(min(curr_frame / frame_count, 1.0))

            cap.release()
            out.release()
            
            status_text.success("SUCCESS: ANALYSIS_COMPLETE")
            
            # 분석 완료 비디오 다운로드 링크 제공
            with open(output_path, "rb") as file:
                st.download_button(
                    label="DOWNLOAD_PROCESSED_VIDEO",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
