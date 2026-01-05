import streamlit as st
import uuid
import os
import subprocess
import sys
from PIL import Image

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Handwriting Generator (Chinese Version)",
    layout="wide"
)

# ======================
# SESSION STATE
# ======================
if "user_id" not in st.session_state:
    st.session_state.user_id = uuid.uuid4().int

if "generated" not in st.session_state:
    st.session_state.generated = False

user_id = st.session_state.user_id

# ======================
# UI HEADER
# ======================
st.markdown("# Handwritten Text Generation (Chinese Version)")
st.markdown("CS172 Final Project")

col1, col2 = st.columns([1, 1])

# ======================
# LEFT: STYLE UPLOAD + GENERATE
# ======================
with col1:
    st.markdown("## Generation")

    uploaded_files = st.file_uploader(
        "Upload at least 15 handwriting reference images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload at least 15 handwriting reference images"
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} style images uploaded")
        st.image(
            [Image.open(f) for f in uploaded_files[:10]],
            caption=[f.name for f in uploaded_files[:10]],
            width=64
        )

    if st.button("Generate", type="primary", use_container_width=True):

        # ---------- 校验 ----------
        if uploaded_files is None or len(uploaded_files) < 15:
            st.error("Please upload at least 15 style reference images")
            st.stop()

        if not os.path.exists("run.py"):
            st.error("run.py not found in current directory")
            st.stop()

        # ---------- 保存 style ----------
        style_dir = f"style_samples/user_{user_id}"
        os.makedirs(style_dir, exist_ok=True)

        for file in uploaded_files:
            save_path = os.path.join(style_dir, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

        # ---------- 日志显示 ----------
        st.markdown("### Generating Status")
        log_placeholder = st.empty()

        log_text = ""

        # ---------- 调用 run.py ----------
        process = subprocess.Popen(
            [
                sys.executable,
                "run.py",
                "--user_id",
                str(user_id)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        with st.spinner("Generating handwriting..."):
            for line in process.stdout:
                last_line = line.strip()
                if last_line:
                    log_placeholder.text(f"▶ {last_line}")

            process.wait()

        if process.returncode != 0:
            st.error("Generation failed")
            st.stop()

        st.success("Font generation completed!")
        st.session_state.generated = True

# ======================
# RIGHT: QUERY
# ======================
with col2:
    st.markdown("## Query")

    text_input = st.text_area(
        "Text to generate:",
        height=100,
        placeholder="请输入中文文本"
    )

    if st.session_state.generated and text_input.strip():
        st.markdown("### Generated Characters")

        font_dir = f"ornamented/user_{user_id}"

        chars = list(text_input.strip())
        images = []
        captions = []

        for ch in chars:
            img_path = os.path.join(font_dir, f"{ch}.png")
            if os.path.exists(img_path):
                images.append(Image.open(img_path))
                captions.append(ch)

        if images:
            st.image(images, caption=captions, width=120)
        else:
            st.warning("No characters found in generated font.")

# ======================
# FOOTER
# ======================
st.divider()
st.markdown("Handwriting Generation System")
