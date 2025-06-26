
import streamlit as st
from PIL import Image
import pandas as pd
import io, datetime
import google.generativeai as genai

# 设置页面
st.set_page_config(page_title="Gemini 图文识别", layout="wide")
st.title("🧠 Gemini 图文识别工具（支持多张图片）")

# 读取 Gemini API 密钥
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-pro-vision')

# 保存所有识别结果
all_results = []

uploaded_files = st.file_uploader("📤 上传多张图片（jpg/png）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for i, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        st.image(image, caption=f"预览图 {i+1}", use_container_width=True)

        with st.spinner(f"正在使用 Gemini 识别第 {i+1} 张图片..."):
            response = model.generate_content([
                "请识别这张图中所有的中文文字内容，以清晰的行排列列出，不要重复和解释。",
                image
            ])
            text = response.text.strip()
            lines = [line for line in text.split("\n") if line.strip()]
            df = pd.DataFrame(lines, columns=["识别结果"])
            all_results.append(df)

    if all_results:
        full_df = pd.concat(all_results, ignore_index=True)
        st.subheader("📝 全部识别结果（可手动修改）")
        edited = st.data_editor(full_df, num_rows="dynamic", use_container_width=True)

        if st.button("📥 导出 Excel"):
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = io.BytesIO()
            edited.to_excel(out, index=False)
            st.download_button("下载文件", out.getvalue(),
                               file_name=f"Gemini识别结果_{now}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success("✅ Excel 已生成，可下载！")
    