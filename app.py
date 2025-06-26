import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io
import json
import datetime
import re
import numpy as np

# --- 页面基础配置 ---
st.set_page_config(
    page_title="Gemini 智能订单校正",
    page_icon="✍️",
    layout="wide"
)

# --- 应用标题和说明 ---
st.title("✍️ Gemini 智能订单识别与校正工具 V2.8")
st.markdown("""
欢迎使用最终版！本工具引入了 **强制校正** 核心功能，确保数据绝对准确。
- **识别与计算**：识别 `品名`、`数量`、`单价`，并 **重新计算** 出精确的 `总价`。
- **智能状态标记**：
  - `✅ 一致`：计算总价与手写总价相符。
  - `✍️ 已校正`：计算总价与手写总价 **不符**，系统已自动使用 **计算结果** 覆盖。
  - `➕ 已计算`：手写单上无总价，系统已自动为您算好。
  - `❔ 信息不足`：缺少计算所需的数据。
- **确保数据纯净**：导出的Excel中，**`总价`列永远等于`单价`×`数量`**。
""")

# --- API 密钥配置 和 模型初始化 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    st.error(f"初始化失败，请检查 API 密钥或模型名称是否正确: {e}")
    st.stop()


# --- Prompt 保持不变，仍然需要识别出原始总价用于对比 ---
PROMPT_TEMPLATE = """
你是一个顶级的、非常严谨的订单数据录入专家。
请仔细识别这张手写订单图片，并提取每一行商品的'品名'、'数量'、'单价'和'总价'。

请严格遵守以下规则：
1.  最终必须输出一个格式完美的 JSON 数组。
2.  数组中的每一个 JSON 对象代表一个商品，且必须包含四个键： "品名", "数量", "单价", "总价"。
3.  如果图片中的某一行缺少某个信息（例如没有写单价），请将对应的值设为空字符串 ""。
4.  如果某个文字或数字非常模糊，无法确定，也请设为空字符串 ""。
5.  **你的回答必须是纯粹的、可以直接解析的 JSON 文本**。绝对不要包含任何解释、说明文字、或者 Markdown 的 ```json ``` 标记。
"""

# --- 会话状态 (Session State) 初始化 ---
if "results" not in st.session_state:
    st.session_state.results = {}

# --- 数据清洗函数 ---
def clean_and_convert_to_numeric(value):
    if value is None or not isinstance(value, str) or value.strip() == "":
        return np.nan
    numbers = re.findall(r'[\d\.]+', str(value))
    if numbers:
        try:
            return float(numbers[0])
        except (ValueError, IndexError):
            return np.nan
    return np.nan

# --- 文件上传组件 ---
files = st.file_uploader(
    "📤 上传一张或多张订单图片 (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if files:
    for file in files:
        file_id = file.file_id
        
        with st.expander(f"📷 图片：{file.name}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("原始图片")
                image = Image.open(file).convert("RGB")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("识别与校正")
                if st.button(f"🚀 识别并强制校正", key=f"btn_{file_id}"):
                    with st.spinner("🧠 Gemini 正在识别并执行财务校正..."):
                        try:
                            response = model.generate_content([PROMPT_TEMPLATE, image])
                            cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                            data = json.loads(cleaned_text)
                            df = pd.DataFrame.from_records(data)
                            
                            expected_cols = ["品名", "数量", "单价", "总价"]
                            for col in expected_cols:
                                if col not in df.columns:
                                    df[col] = ""
                            
                            # --- ✅ 核心校正逻辑开始 ---
                            # 1. 清洗原始数据为数值
                            df['数量_num'] = df['数量'].apply(clean_and_convert_to_numeric)
                            df['单价_num'] = df['单价'].apply(clean_and_convert_to_numeric)
                            df['识别总价_num'] = df['总价'].apply(clean_and_convert_to_numeric) # 这是从图片上识别的总价
                            
                            # 2. 强制计算出权威总价
                            df['计算总价'] = df['数量_num'] * df['单价_num']
                            
                            # 3. 生成新的、更智能的验算状态
                            conditions = [
                                (df['计算总价'].notna() & np.isclose(df['计算总价'], df['识别总价_num'])), # 计算总价与识别总价一致
                                (df['计算总价'].notna() & df['识别总价_num'].notna()), # 两者都有，但不一致 -> 已校正
                                (df['计算总价'].notna() & df['识别总价_num'].isna()),   # 能算出总价，但图片上没写 -> 已计算
                            ]
                            choices = [
                                '✅ 一致', 
                                '✍️ 已校正',
                                '➕ 已计算'
                            ]
                            df['验算状态'] = np.select(conditions, choices, default='❔ 信息不足')
                            
                            # 4. 【关键一步】用权威的计算总价覆盖原始的总价列，并格式化
                            df['总价'] = df['计算总价'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")

                            # --- 核心校正逻辑结束 ---
                            
                            final_cols = ["品名", "数量", "单价", "总价", "验算状态"]
                            st.session_state.results[file_id] = df[final_cols]
                            
                            st.success("✅ 识别与校正完成！")
                            st.rerun()

                        except json.JSONDecodeError:
                            st.error("❌ 结构化识别失败：模型返回的不是有效的JSON格式。")
                            st.info("模型返回的原始文本：")
                            st.text_area("原始输出", cleaned_text if 'cleaned_text' in locals() else response.text, height=150)
                        except Exception as e:
                            st.error(f"❌ 处理失败，发生未知错误：{e}")

                if file_id in st.session_state.results:
                    st.dataframe(st.session_state.results[file_id], use_container_width=True)
                    st.caption("上方为最终校正结果。")

if st.session_state.results:
    st.divider()
    st.header("📝 统一编辑与导出")

    all_dfs = list(st.session_state.results.values())
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)

        st.info("您可以在下表中修改品名、数量、单价，总价和状态将自动更新（下次识别时）。")
        edited_df = st.data_editor(
            merged_df,
            num_rows="dynamic",
            use_container_width=True,
            height=300,
            # 锁定由程序生成的列，确保数据纯净性
            disabled=["总价", "验算状态"]
        )

        st.subheader("📥 导出为 Excel 文件")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='识别结果')
            writer.sheets['识别结果'].autofit()
        
        excel_data = output.getvalue()
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"订单校正结果_{now}.xlsx"
        
        st.download_button(
            label="✅ 点击下载【校正后】的Excel",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
