import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import time
from PIL import Image, ImageOps 

# [IMPORT MOVIEPY] Dengan penanganan error versi
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError:
    try:
        from moviepy import VideoFileClip, AudioFileClip
    except ImportError:
        st.error("Library moviepy belum terinstall. Jalankan: pip install moviepy")

# ==========================================
# 1. KONFIGURASI HALAMAN & STATE
# ==========================================
st.set_page_config(
    page_title="Video Processing Studio",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi Session State
if 'webcam_video_path' not in st.session_state:
    st.session_state['webcam_video_path'] = None

# ==========================================
# 2. CUSTOM CSS (PROFESSIONAL THEME)
# ==========================================
st.markdown("""
    <style>
    /* --- GLOBAL SETTINGS --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC; /* Slate 50 */
    }
    
    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.02);
    }
    
    /* --- HEADINGS --- */
    h1 {
        color: #1E3A8A; /* Dark Blue */
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #334155; /* Slate 700 */
        font-weight: 700;
    }
    
    /* --- METRICS CARD --- */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #3B82F6; /* Accent Border */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }
    label[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-weight: 700;
    }
    
    /* --- BUTTONS --- */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Secondary Button */
    div.stButton > button[kind="secondary"] {
        background: #FFFFFF;
        color: #334155;
        border: 1px solid #CBD5E1;
        box-shadow: none;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #F8FAFC;
        border-color: #94A3B8;
    }

    /* --- PREVIEW HEADER LABELS --- */
    .preview-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #475569;
        margin-bottom: 0.8rem;
        font-weight: 700;
        background: #E2E8F0;
        padding: 5px 10px;
        border-radius: 4px;
        display: inline-block;
    }
    
    /* --- LANDING PAGE STYLES --- */
    .hero-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 5px;
        color: white !important;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 500;
        color: #DBEAFE !important;
        margin-bottom: 15px;
    }
    .hero-desc {
        font-size: 1rem;
        color: #E2E8F0;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .landing-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .landing-card:hover {
        transform: translateY(-5px);
        border-color: #3B82F6;
    }
    .step-number {
        background-color: #EFF6FF;
        color: #2563EB;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 0 auto 15px auto;
    }
    
    /* --- ACTION BOX --- */
    .action-box {
        background-color: #F0F9FF; /* Light Blue */
        border: 2px dashed #0EA5E9; /* Sky Blue Border */
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        margin-top: 40px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .action-box:hover {
        background-color: #E0F2FE;
    }
    .action-title {
        color: #0284C7;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    /* --- UPLOAD BOX --- */
    .stFileUploader {
        border: 2px dashed #93C5FD;
        border-radius: 12px;
        padding: 20px;
        background-color: #F8FAFC;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI LOGIKA (CORE LOGIC)
# ==========================================
def process_single_frame(frame, blur_type, blur_amount, flip_type, use_watermark, watermark_text, logo_img):
    try:
        # 1. FLIP
        if flip_type == "Horizontal":
            frame = cv2.flip(frame, 1)
        elif flip_type == "Vertical":
            frame = cv2.flip(frame, 0)
        elif flip_type == "Both":
            frame = cv2.flip(frame, -1)

        # 2. BLUR
        k_size = int(blur_amount) if int(blur_amount) % 2 == 1 else int(blur_amount) + 1
        
        if blur_type == "Average":
            frame = cv2.blur(frame, (k_size, k_size))
        elif blur_type == "Gaussian":
            frame = cv2.GaussianBlur(frame, (k_size, k_size), 0)
        elif blur_type == "Median":
            frame = cv2.medianBlur(frame, k_size)

        # 3. LOGO OVERLAY
        if logo_img is not None:
            h, w, _ = frame.shape
            l_h, l_w, _ = logo_img.shape
            scale_factor = (w * 0.15) / l_w
            new_size = (int(l_w * scale_factor), int(l_h * scale_factor))
            
            if new_size[0] > 0 and new_size[1] > 0:
                logo_resized = cv2.resize(logo_img, new_size)
                rh, rw, _ = logo_resized.shape
                y_off = 20
                x_off = w - rw - 20
                if y_off + rh < h and x_off + rw < w:
                    frame[y_off:y_off+rh, x_off:x_off+rw] = logo_resized

        # 4. TEXT WATERMARK
        if use_watermark and watermark_text:
            cv2.putText(frame, watermark_text, (32, 52), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, watermark_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    except Exception:
        pass
    return frame

# [FUNGSI AUDIO MERGE]
def merge_audio(video_path, audio_source_path, output_path):
    try:
        video_clip = VideoFileClip(video_path)
        original_clip = VideoFileClip(audio_source_path)
        
        if original_clip.audio:
            final_clip = video_clip.set_audio(original_clip.audio)
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            video_clip.close()
            original_clip.close()
            final_clip.close()
            return True
        else:
            video_clip.close()
            original_clip.close()
            return False
    except Exception as e:
        print(f"Error merging audio: {e}")
        return False

def main():
    # --- SIDEBAR: CONFIGURATION ---
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        st.divider()
        
        # 1. Input Source
        st.markdown("**1. Source Selection**")
        source_option = st.selectbox(
            "Select Method", 
            ["Upload File", "Webcam Recording"],
            label_visibility="collapsed"
        )

        input_path = None
        
        if source_option == "Upload File":
            if st.session_state['webcam_video_path']:
                st.session_state['webcam_video_path'] = None
                st.rerun()
            
            uploaded_file = st.file_uploader("Drop video file here (.mp4, .avi)", type=['mp4', 'avi'])
            if uploaded_file:
                # [PENTING] Suffix .mp4 agar dikenali MoviePy
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                temp_file.close() # Close handle immediately for Windows safety
                input_path = temp_file.name

        else: # Webcam
            if st.session_state['webcam_video_path'] is None:
                st.info("Set duration and click start.")
                duration = st.slider("Duration (sec)", 5, 30, 10)
                if st.button("🔴 Start Recording"):
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("Cannot open webcam.")
                    else:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = 30.0
                        
                        t_rec = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        t_rec.close() # Close handle
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(t_rec.name, fourcc, fps, (width, height))
                        
                        st_image = st.empty()
                        progress_bar = st.progress(0)
                        
                        start_time = time.time()
                        while int(time.time() - start_time) < duration:
                            ret, frame = cap.read()
                            if not ret: break
                            out.write(frame)
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st_image.image(frame_rgb, channels="RGB", use_container_width=True)
                            elapsed = time.time() - start_time
                            progress_bar.progress(min(elapsed / duration, 1.0))
                        
                        cap.release()
                        out.release()
                        st.session_state['webcam_video_path'] = t_rec.name
                        st.success("Recording saved.")
                        time.sleep(1)
                        st.rerun()
            else:
                st.success("✅ Webcam video loaded.")
                input_path = st.session_state['webcam_video_path']
                if st.button("Reset / Record Again", type="secondary"):
                    st.session_state['webcam_video_path'] = None
                    st.rerun()

        st.divider()

        # 2. Effects Configuration
        st.markdown("**2. Effect Settings**")
        
        with st.expander("🎨 Color & Filters", expanded=True):
            use_gray = st.toggle("Grayscale Mode")
            blur_type = st.selectbox("Blur Type", ["None", "Average", "Gaussian", "Median"])
            blur_amt = st.slider("Intensity", 1, 21, 1, step=2, disabled=(blur_type=="None"))

        with st.expander("📐 Geometry & Orientation"):
            flip_type = st.selectbox("Flip Mode", ["None", "Horizontal", "Vertical", "Both"])

        with st.expander("💧 Branding & Overlay"):
            use_watermark = st.toggle("Enable Watermark")
            wm_text = ""
            if use_watermark:
                wm_text = st.text_input("Watermark Text", "FKIP ULM")
            
            st.markdown("Logo Overlay")
            logo_up = st.file_uploader("Upload Logo (PNG)", type=['png', 'jpg'], key="logo")
            logo_cv = None
            if logo_up:
                file_bytes = np.asarray(bytearray(logo_up.read()), dtype=np.uint8)
                logo_cv = cv2.imdecode(file_bytes, 1)

    # --- MAIN CONTENT AREA ---
    
    if input_path is not None:
        # HEADER DASHBOARD
        st.markdown("### 🎬 Studio Workspace")
        st.markdown("---")

        # Metadata
        cap_temp = cv2.VideoCapture(input_path)
        vid_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap_temp.get(cv2.CAP_PROP_FPS)
        vid_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_duration = vid_frames / vid_fps if vid_fps > 0 else 0
        
        ret, preview_frame = cap_temp.read()
        cap_temp.release()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Resolution", f"{vid_width} x {vid_height}")
        m2.metric("FPS", f"{vid_fps:.1f}")
        m3.metric("Total Frames", f"{vid_frames}")
        mins, secs = divmod(vid_duration, 60)
        m4.metric("Duration", f"{int(mins)}m {int(secs)}s")

        st.markdown("<br>", unsafe_allow_html=True) 

        # Preview Row
        c_orig, c_res = st.columns(2)
        
        with c_orig:
            st.markdown("<div class='preview-header'>Original Input</div>", unsafe_allow_html=True)
            if preview_frame is not None:
                frame_ori_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_ori_rgb, use_container_width=True)

        with c_res:
            st.markdown("<div class='preview-header'>Processed Preview</div>", unsafe_allow_html=True)
            if preview_frame is not None:
                proc_frame = preview_frame.copy()
                if use_gray:
                    proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
                    proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_GRAY2BGR)
                
                proc_frame = process_single_frame(
                    proc_frame, blur_type, blur_amt, flip_type, 
                    use_watermark, wm_text, logo_cv
                )
                frame_res_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_res_rgb, use_container_width=True)

        st.divider()
        
        col_exec_1, col_exec_2 = st.columns([3, 1])
        with col_exec_1:
            st.subheader("⚙️ Processing Engine")
            st.markdown("Start rendering to apply all effects, view system logs, and download the final result.")
        with col_exec_2:
            show_terminal = st.toggle("Show Terminal Log", value=True)
        
        if st.button("🚀 START RENDERING PROCESS", type="primary"):
            st.markdown("---")
            c_term, c_prog = st.columns([2, 1], gap="large")
            
            with c_term:
                st.markdown("**System Terminal Output:**")
                terminal_placeholder = st.empty()
                st.markdown("**Executed Code Snippet (Educational):**")
                code_placeholder = st.empty() 
            
            with c_prog:
                st.markdown("**Real-time Monitor:**")
                video_display = st.empty()
                st.write("") 
                st.markdown("**Status:**")
                prog_bar = st.progress(0)
                stat_text = st.empty()
                frame_metric = st.empty()

            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # File Temporary untuk Video BISU
            t_out_silent = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_out_silent.close() # Close handle immediately
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(t_out_silent.name, fourcc, fps, (width, height))
            
            log_history = [] 
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Logic & Educational Code Generation
                current_process_steps = []
                educational_code = []
                educational_code.append("# 1. Read Frame\nret, frame = cap.read()")
                
                if use_gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    current_process_steps.append("cv2.cvtColor(BGR2GRAY)")
                    educational_code.append("# 2. Grayscale\nframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\nframe = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)")
                
                final_frame = process_single_frame(frame, blur_type, blur_amt, flip_type, use_watermark, wm_text, logo_cv)
                
                if flip_type != "None": 
                    current_process_steps.append(f"cv2.flip('{flip_type}')")
                    flip_code = 1 if flip_type == "Horizontal" else (0 if flip_type == "Vertical" else -1)
                    educational_code.append(f"# 3. Flip\nframe = cv2.flip(frame, {flip_code})")
                    
                if blur_type != "None": 
                    current_process_steps.append(f"cv2.{blur_type}(k={blur_amt})")
                    k = int(blur_amt) if int(blur_amt)%2==1 else int(blur_amt)+1
                    educational_code.append(f"# 4. Blur ({blur_type})\nframe = cv2.blur/GaussianBlur/medianBlur...")

                if use_watermark: 
                    current_process_steps.append("cv2.putText()")
                    educational_code.append(f"# 5. Watermark\ncv2.putText(frame, '{wm_text}', ...)")
                
                if logo_cv is not None: 
                    current_process_steps.append("Overlay(Logo)")
                    educational_code.append("# 6. Overlay\nframe[y:y+h, x:x+w] = logo_resized")
                
                steps_str = " -> ".join(current_process_steps) if current_process_steps else "No Operation"
                timestamp = time.strftime("%H:%M:%S")
                
                log_line = f"<span style='color:#00ff00'>[{timestamp}]</span> <span style='color:#facc15'>[FRAME {frame_count:04d}]</span> {steps_str}"
                log_history.append(log_line)
                
                out.write(final_frame)
                
                if frame_count % 3 == 0:
                    frame_disp = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    video_display.image(frame_disp, channels="RGB", use_container_width=True, caption=f"Processing Frame: {frame_count}")

                if show_terminal and frame_count % 2 == 0:
                    log_content = "<br>".join(log_history)
                    terminal_html = f"""
                    <div style="height: 200px; overflow-y: auto; background-color: #0F172A; color: #E2E8F0; font-family: 'Courier New', monospace; font-size: 12px; padding: 10px; border-radius: 5px; border: 1px solid #334155; display: flex; flex-direction: column-reverse;">
                        <div style="margin-top: auto;">{log_content}</div>
                    </div>
                    """
                    terminal_placeholder.markdown(terminal_html, unsafe_allow_html=True)
                    full_code_text = "\n\n".join(educational_code)
                    code_placeholder.code(full_code_text, language='python')
                
                frame_count += 1
                if total_frames > 0:
                    prog_bar.progress(min(frame_count / total_frames, 1.0))
                    stat_text.text(f"Processing: {int((frame_count/total_frames)*100)}%")
                    frame_metric.metric("Frames", f"{frame_count}/{total_frames}")

            cap.release()
            out.release()
            
            # --- AUDIO MERGING LOGIC ---
            stat_text.text("Merging Audio from Source... (Please wait)")
            final_output_path = t_out_silent.name # Default backup
            
            if source_option == "Upload File":
                t_out_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                t_out_audio.close() # Close immediately
                
                success = merge_audio(t_out_silent.name, input_path, t_out_audio.name)
                
                if success:
                    final_output_path = t_out_audio.name
                    log_history.append(f"<br><span style='color:#3b82f6'><b>[SYSTEM] Audio Merged Successfully!</b></span>")
                else:
                    log_history.append(f"<br><span style='color:#f59e0b'><b>[SYSTEM] No audio track found. Returning silent video.</b></span>")
            else:
                log_history.append(f"<br><span style='color:#f59e0b'><b>[SYSTEM] Webcam (OpenCV) is video-only.</b></span>")

            # Final Log
            log_history.append(f"<br><span style='color:#3b82f6'><b>[{time.strftime('%H:%M:%S')}] [SYSTEM] PROCESS COMPLETE.</b></span>")
            final_log_content = "<br>".join(log_history)
            final_html = f"""
            <div style="height: 200px; overflow-y: auto; background-color: #0F172A; color: #E2E8F0; font-family: 'Courier New', monospace; font-size: 12px; padding: 10px; border-radius: 5px; border: 1px solid #334155;">
                {final_log_content}
            </div><br>
            """
            terminal_placeholder.markdown(final_html, unsafe_allow_html=True)
            stat_text.success("Selesai!")

            st.divider()
            st.subheader("📥 Export Result")
            
            with open(final_output_path, 'rb') as f:
                st.download_button(
                    label="DOWNLOAD VIDEO MP4",
                    data=f,
                    file_name="Processed_Video_Final.mp4",
                    mime="video/mp4",
                    type="primary"
                )
    
    # ==========================================
    # 4. TAMPILKAN LANDING PAGE (JIKA BELUM ADA INPUT)
    # ==========================================
    else:
        # --- 1. ILUSTRASI LOKAL (3 KOLOM KECIL) ---
        col_img1, col_img2, col_img3 = st.columns(3)
        local_img_paths = ["img1.jpeg", "img2.jpeg", "img3.jpeg"] 
        
        def load_local_img(path, col):
            with col:
                if os.path.exists(path):
                    try:
                        img = Image.open(path)
                        # Fix size using ImageOps.fit
                        img_fixed = ImageOps.fit(img, (600, 400), method=Image.Resampling.LANCZOS)
                        st.image(img_fixed, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning(f"File '{path}' not found.")
                    st.image("https://via.placeholder.com/600x400?text=Image+Not+Found", use_container_width=True)

        load_local_img(local_img_paths[0], col_img1)
        load_local_img(local_img_paths[1], col_img2)
        load_local_img(local_img_paths[2], col_img3)

        st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Video Processing Studio</div>
            <div class="hero-subtitle">FKIP ULM - Computer Education Department</div>
            <div class="hero-desc">
                Sebuah aplikasi komprehensif untuk memproses video digital menggunakan kekuatan Python & OpenCV.
                Dilengkapi dengan fitur real-time monitoring, efek visual, dan terminal edukasi.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Workflow Aplikasi")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="landing-card">
                <div class="step-number">1</div>
                <h4>Select Source</h4>
                <p style="font-size: 0.9rem; color: #64748B;">
                    Gunakan panel sidebar untuk mengupload video atau merekam webcam.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="landing-card">
                <div class="step-number">2</div>
                <h4>Configure</h4>
                <p style="font-size: 0.9rem; color: #64748B;">
                    Aktifkan efek Grayscale, Blur, Flip, atau tambahkan Watermark & Logo.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="landing-card">
                <div class="step-number">3</div>
                <h4>Process</h4>
                <p style="font-size: 0.9rem; color: #64748B;">
                    Klik Start untuk melihat proses frame-by-frame dan log sistem secara real-time.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="landing-card">
                <div class="step-number">4</div>
                <h4>Export</h4>
                <p style="font-size: 0.9rem; color: #64748B;">
                    Unduh hasil video yang telah diproses dalam format MP4 berkualitas tinggi.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <div class="action-box">
            <div class="action-title">⬅️ Mulai dari Panel Kiri</div>
            <p style="color: #64748B; font-size: 1.1rem;">
                Silakan pilih <b>Upload File</b> atau <b>Webcam Recording</b> pada Sidebar untuk membuka Dashboard Editor.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()