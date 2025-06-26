import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io, datetime

st.set_page_config(page_title="Gemini 手写识别", layout="wide")
st.title("🧠 Gemini OCR：手写进货单识别工具")

# 加载 API key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# 初始化模型
model = genai.GenerativeModel("gemini-pro-vision")

# 全局保存识别结果
if "results" not in st.session_state:
    st.session_state.results = []

# 多文件上传
files = st.file_uploader("📤 上传多张图片（jpg/png）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if files:
    for i, file in enumerate(files):
        with st.expander(f"📷 第{i+1}张图片：{file.name}", expanded=False):
            image = Image.open(file).convert("RGB")
            st.image(image, caption=file.name, use_container_width=True)

            if st.button(f"🚀 识别第{i+1}张", key=f"btn_{i}"):
                with st.spinner("Gemini 正在识别中..."):
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    img_bytes = buf.getvalue()

                    response = model.generate_content([
                        {"mime_type": "image/jpeg", "data": img_bytes},
                        "请识别这张图中的所有中文手写内容，以自然分行输出，不要解释。"
                    ])

                    text = response.text.strip()
                    lines = text.split("\n")
                    df = pd.DataFrame(lines, columns=["识别结果"])
                    st.session_state.results.append(df)
                    st.success("✅ 识别完成，可在下方修改")

# 编辑 + 合并所有识别结果
if st.session_state.results:
    st.subheader("📝 可编辑所有识别内容")
    merged_df = pd.concat(st.session_state.results, ignore_index=True)
    edited = st.data_editor(merged_df, num_rows="dynamic", use_container_width=True)

    if st.button("📥 导出所有为 Excel"):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = io.BytesIO()
        edited.to_excel(out, index=False)
        st.download_button("下载识别结果", out.getvalue(),
                           file_name=f"识别结果_{now}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success("✅ 已导出 Excel！")
