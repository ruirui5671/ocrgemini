import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io
import json
import datetime
import re  # 引入正则表达式库，用于数据清洗
import numpy as np # 引入Numpy库，用于更专业地处理数值和空值

# --- 页面基础配置 ---
st.set_page_config(
    page_title="Gemini 智能订单识别",
    page_icon="🧠",
    layout="wide"
)

# --- 应用标题和说明 ---
st.title("✅ Gemini 智能订单识别与验算工具 V2.7")
st.markdown("""
欢迎使用！此版本引入了强大的 **自动验算** 功能。
- **自动提取字段**：识别 `品名`、`数量`、`单价` 和 `总价`。
- **智能交叉验算**：自动计算 `数量 × 单价` 并与识别出的 `总价` 对比，标记出不一致的数据。
- **统一编辑和导出**：所有结果及验算状态将合并在一张表中，供您修改并导出为 Excel。
""")

# --- API 密钥配置 和 模型初始化 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    st.error(f"初始化失败，请检查 API 密钥或模型名称是否正确: {e}")
    st.stop()


# ✅ --- 1. 更新 Prompt，增加 "总价" 字段 ---
PROMPT_TEMPLATE = """
你是一个顶级的、非常严谨的订单数据录入专家。
请仔细识别这张手写订单图片，并提取每一行商品的'品名'、'数量'、'单价'和'总价'。

请严格遵守以下规则：
1.  最终必须输出一个格式完美的 JSON 数组。
2.  数组中的每一个 JSON 对象代表一个商品，且必须包含四个键： "品名", "数量", "单价", "总价"。
3.  如果图片中的某一行缺少某个信息（例如没有写单价），请将对应的值设为空字符串 ""。
4.  如果某个文字或数字非常模糊，无法确定，也请设为空字符串 ""。
5.  **你的回答必须是纯粹的、可以直接解析的 JSON 文本**。绝对不要包含任何解释、说明文字、或者 Markdown 的 ```json ``` 标记。

例如，对于一张包含 "雪花纯生 5箱 85元 425" 和 "青岛原浆 3箱 120元" 的图片，你应该返回：
[
    { "品名": "雪花纯生", "数量": "5", "单价": "85", "总价": "425" },
    { "品名": "青岛原浆", "数量": "3", "单价": "120", "总价": "" }
]
"""

# --- 会话状态 (Session State) 初始化 ---
if "results" not in st.session_state:
    st.session_state.results = {}

# ✅ --- 2. 定义数据清洗函数 ---
def clean_and_convert_to_numeric(value):
    """从字符串中提取数字并转换为浮点数，无法转换则返回NaN"""
    if value is None or not isinstance(value, str) or value.strip() == "":
        return np.nan # 使用Numpy的NaN (Not a Number) 代表缺失的数值
    # 使用正则表达式查找字符串中的第一个数字（可以是整数或小数）
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
                st.subheader("识别与验算")
                if st.button(f"🚀 识别并验算", key=f"btn_{file_id}"):
                    with st.spinner("🧠 最新 Gemini 模型正在识别和验算中..."):
                        try:
                            response = model.generate_content([PROMPT_TEMPLATE, image])
                            cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                            data = json.loads(cleaned_text)
                            df = pd.DataFrame.from_records(data)

                            # --- ✅ 3. 执行验算逻辑 ---
                            # 确保所有需要的列都存在
                            expected_cols = ["品名", "数量", "单价", "总价"]
                            for col in expected_cols:
                                if col not in df.columns:
                                    df[col] = "" # 如果模型没返回，则创建空列
                            
                            # 创建用于计算的临时列，清洗数据
                            df['数量_num'] = df['数量'].apply(clean_and_convert_to_numeric)
                            df['单价_num'] = df['单价'].apply(clean_and_convert_to_numeric)
                            df['总价_num'] = df['总价'].apply(clean_and_convert_to_numeric)
                            
                            # 计算总价
                            df['计算总价'] = df['数量_num'] * df['单价_num']
                            
                            # --- ✅ 4. 生成验算状态 ---
                            conditions = [
                                # 条件1: 计算总价 和 识别总价 在数值上非常接近 (处理浮点数精度问题)
                                (np.isclose(df['计算总价'], df['总价_num'], equal_nan=True)),
                                # 条件2: 两者都能计算，但结果不一致
                                (df['计算总价'].notna() & df['总价_num'].notna()),
                                # 条件3: 能算出总价，但图片上没写总价
                                (df['计算总价'].notna() & df['总价_num'].isna())
                            ]
                            choices = [
                                '✅ 一致', 
                                '⚠️ 不一致',
                                '❔ 待补全'
                            ]
                            df['验算状态'] = np.select(conditions, choices, default='❔ 无法计算')
                            
                            # 准备最终显示的DataFrame
                            final_cols = ["品名", "数量", "单价", "总价", "验算状态"]
                            st.session_state.results[file_id] = df[final_cols]
                            
                            st.success("✅ 识别验算完成！")
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
            height=300,
            # 让用户不能编辑“验算状态”列，因为它是由程序生成的
            disabled=["验算状态"]
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
