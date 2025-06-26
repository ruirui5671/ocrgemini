import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

# 设置 Gemini API 密钥
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# 使用 Gemini 2.5 Pro
model = genai.GenerativeModel("gemini-2.5-pro")

st.set_page_config(page_title="手写OCR by Gemini 2.5 Pro")
st.title("🧠 Gemini 2.5 Pro 手写文字识别")

uploaded_file = st.file_uploader("📤 上传一张手写进货单图片", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上传图片预览", use_container_width=True)

    if st.button("🔍 识别文字"):
        with st.spinner("Gemini 识别中，请稍候..."):
            # 读取图片并转为字节
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()

            try:
                response = model.generate_content([
                    {"mime_type": "image/jpeg", "data": img_bytes},
                    "请识别这张图中的所有中文手写内容，以自然分行输出，不要解释。"
                ])
                st.success("识别完成 ✅")
                st.text_area("识别结果", response.text, height=300)
            except Exception as e:
                st.error(f"调用 Gemini 出错：{e}")
