import streamlit as st
from PIL import Image
import pandas as pd
import io, datetime
import google.generativeai as genai

# 读取密钥（已在 Secrets UI 设置）
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# 选择 Vision 模型
model = genai.GenerativeModel("gemini-pro-vision")

st.set_page_config(page_title="Gemini 图文识别", layout="wide")
st.title("📷 Gemini 图文识别工具（多图批处理）")

uploaded_files = st.file_uploader(
    "上传图片（jpg / png，可多选）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

all_results = []

if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        img = Image.open(file).convert("RGB")
        st.image(img, caption=f"第 {idx+1} 张：{file.name}", use_container_width=True)

        if st.button(f"识别第 {idx+1} 张", key=f"btn_{idx}"):
            with st.spinner("Gemini 正在识别…"):
                # 把 PIL 图像转成 bytes
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                img_bytes = buf.getvalue()

                # 以 Part 格式调用
                response = model.generate_content([
                    {"mime_type": "image/jpeg", "data": img_bytes},
                    "请识别这张图中的所有中文手写内容，以自然分行输出，不要解释。"
                ])
                text = response.text.strip()
                lines = [ln for ln in text.splitlines() if ln.strip()]
                df = pd.DataFrame(lines, columns=["识别结果"])
                all_results.append(df)
                st.success("识别完成！")

    if all_results:
        full_df = pd.concat(all_results, ignore_index=True)
        st.subheader("📝 全部识别结果（可编辑）")
        edited = st.data_editor(full_df, num_rows="dynamic", use_container_width=True)

        if st.button("📥 导出 Excel"):
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = io.BytesIO()
            edited.to_excel(out, index=False)
            st.download_button(
                "下载 Excel",
                data=out.getvalue(),
                file_name=f"Gemini_OCR_{now}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.success("✅ 已生成 Excel")
