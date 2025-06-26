import streamlit as st
from PIL import Image
import pandas as pd
import io, datetime
import google.generativeai as genai

# 设置页面配置
st.set_page_config(page_title="Gemini 2.5 Pro 图文识别", layout="wide")
st.title("📷 Gemini 2.5 Pro 图文识别工具")

# 配置 Gemini API 密钥
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# 初始化 Gemini 2.5 Pro 模型
model = genai.GenerativeModel('gemini-2.5-pro')

# 文件上传
uploaded_files = st.file_uploader("上传图片（支持多张）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    all_results = []
    for i, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        st.image(image, caption=f"预览图 {i+1}", use_container_width=True)

        with st.spinner(f"正在识别第 {i+1} 张图片..."):
            # 将图像转换为适合模型输入的格式
            image_data = genai.types.Blob(
                mime_type="image/jpeg",
                data=file.getvalue()
            )
            # 调用模型进行内容生成
            response = model.generate_content([
                "请识别这张图中的所有中文文字内容，并以清晰的行排列列出，不要重复和解释。",
                image_data
            ])
            text = response.text.strip()
            lines = [line for line in text.split("\n") if line.strip()]
            df = pd.DataFrame(lines, columns=["识别结果"])
            all_results.append(df)

    if all_results:
        full_df = pd.concat(all_results, ignore_index=True)
        st.subheader("📝 识别结果（可编辑）")
        edited = st.data_editor(full_df, num_rows="dynamic", use_container_width=True)

        if st.button("📥 导出为 Excel"):
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = io.BytesIO()
            edited.to_excel(out, index=False)
            st.download_button("下载文件", out.getvalue(),
                               file_name=f"识别结果_{now}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success("✅ Excel 文件已生成，可下载！")
