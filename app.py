import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io
import json
import datetime

# --- 页面基础配置 ---
st.set_page_config(
    page_title="Gemini 智能订单识别",
    page_icon="🧠",
    layout="wide"
)

# --- 应用标题和说明 ---
st.title("🚀 Gemini 最新模型: 智能手写订单识别工具 V2.6")
st.markdown("""
欢迎使用！本工具已搭载 Google **当前最新、最强大的 `Gemini 1.5 Pro` 模型**，为您提供顶级的识别体验。
- **自动提取字段**：上传手写订单图片，将自动识别出 `品名`、`数量`、`单价` 等关键信息。
- **统一编辑和导出**：所有识别结果会合并在一张表格中，您可以方便地进行修改、补充，并一键导出为 Excel 文件。
""")

# --- API 密钥配置 和 模型初始化 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    st.error(f"初始化失败，请检查 API 密钥或模型名称是否正确: {e}")
    st.stop()


# --- Gemini 的指令 (Prompt)，针对新模型优化 ---
PROMPT_TEMPLATE = """
你是一个顶级的订单数据录入专家。
请严格按照图片中的手写内容，识别并提取每一行商品的'品名'、'数量'和'单价'。

请严格遵守以下规则：
1.  最终必须输出一个格式完美的 JSON 数组。
2.  数组中的每一个 JSON 对象代表一个商品，且必须包含三个键： "品名", "数量", "单价"。
3.  如果图片中的某一行缺少某个信息（例如没有写单价或数量），请将对应的值设为空字符串 ""。
4.  如果某个文字或数字非常模糊，无法确定，也请设为空字符串 ""。
5.  **你的回答必须是纯粹的、可以直接解析的 JSON 文本**。绝对不要包含任何解释、说明文字、或者 Markdown 的 ```json ``` 标记。

例如，对于一张包含 "雪花纯生 5箱 85" 和 "青岛原浆 3箱" 的图片，你应该返回：
[
    { "品名": "雪花纯生", "数量": "5", "单价": "85" },
    { "品名": "青岛原浆", "数量": "3", "单价": "" }
]
"""

# --- 会话状态 (Session State) 初始化 ---
if "results" not in st.session_state:
    st.session_state.results = {}

# --- 文件上传组件 ---
files = st.file_uploader(
    "📤 上传一张或多张订单图片 (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if files:
    for file in files:
        # ✅ --- 关键修正点：使用 file.file_id 作为唯一的 key ---
        file_id = file.file_id
        
        with st.expander(f"📷 图片：{file.name}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("原始图片")
                image = Image.open(file).convert("RGB")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("识别与处理")
                if st.button(f"🚀 使用最新模型识别", key=f"btn_{file_id}"):
                    with st.spinner("🧠 最新 Gemini 模型正在全力识别中..."):
                        try:
                            response = model.generate_content([PROMPT_TEMPLATE, image])
                            cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                            data = json.loads(cleaned_text)
                            df = pd.DataFrame.from_records(data)

                            expected_cols = ["品名", "数量", "单价"]
                            for col in expected_cols:
                                if col not in df.columns:
                                    df[col] = ""
                            
                            st.session_state.results[file_id] = df[expected_cols]
                            st.success("✅ 识别完成！")
                            st.rerun()

                        except json.JSONDecodeError:
                            st.error("❌ 结构化识别失败：模型返回的不是有效的JSON格式。")
                            st.info("模型返回的原始文本：")
                            st.text_area("原始输出", cleaned_text if 'cleaned_text' in locals() else response.text, height=150)
                        except Exception as e:
                            st.error(f"❌ 处理失败，发生未知错误：{e}")

                if file_id in st.session_state.results:
                    st.dataframe(st.session_state.results[file_id], use_container_width=True)
                    st.caption("上方为识别结果。如需重新识别，请再次点击识别按钮。")

if st.session_state.results:
    st.divider()
    st.header("📝 统一编辑与导出")

    all_dfs = list(st.session_state.results.values())
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)

        st.info("您可以在下表中直接修改或添加行。所有修改将一并导出。")
        edited_df = st.data_editor(
            merged_df,
            num_rows="dynamic",
            use_container_width=True,
            height=300
        )

        st.subheader("📥 导出为 Excel 文件")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='识别结果')
            writer.sheets['识别结果'].autofit()
        
        excel_data = output.getvalue()
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"订单识别结果_{now}.xlsx"
        
        st.download_button(
            label="✅ 点击这里下载 Excel",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
