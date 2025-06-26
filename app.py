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
    page_title="Gemini 并列诊断",
    page_icon="📊",
    layout="wide"
)

# --- 应用标题和说明 ---
st.title("📊 Gemini 智能订单诊断工具 V3.1 (并列诊断版)")
st.markdown("""
欢迎使用！本工具通过 **并列展示** 计算结果与推算结果，让您对订单数据一目了然，快速定位潜在错误。
- **忠实识别**：完整展示图片中的 `识别数量`、`识别单价`、`识别总价`。
- **并列对比**：
  - **`计算总价`**: `识别数量 × 识别单价` 的结果。
  - **`[按总价]推算数量`**: 假设总价和单价正确，反推出的数量。
  - **`[按总价]推算单价`**: 假设总价和数量正确，反推出的单价。
- **一目了然**：通过直接对比这几列数字，您可以瞬间判断问题所在。
""")

# --- API 密钥配置 和 模型初始化 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    st.error(f"初始化失败，请检查 API 密钥或模型名称是否正确: {e}")
    st.stop()


# --- Prompt 保持不变，它负责抓取最原始的数据 ---
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
            st.image(Image.open(file).convert("RGB"), width=250) # 图片放小一点，给表格留出空间
            
            if st.button(f"🚀 开始并列诊断", key=f"btn_{file_id}"):
                with st.spinner("🕵️ Gemini 正在进行识别和并列诊断..."):
                    try:
                        image = Image.open(file).convert("RGB") # 重新打开图片用于识别
                        response = model.generate_content([PROMPT_TEMPLATE, image])
                        cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                        data = json.loads(cleaned_text)
                        
                        df = pd.DataFrame.from_records(data)
                        df.rename(columns={
                            "数量": "识别数量",
                            "单价": "识别单价",
                            "总价": "识别总价"
                        }, inplace=True)
                        
                        # --- ✅ 核心诊断逻辑开始 ---
                        expected_cols = ["品名", "识别数量", "识别单价", "识别总价"]
                        for col in expected_cols:
                            if col not in df.columns:
                                df[col] = ""
                        
                        df['数量_num'] = df['识别数量'].apply(clean_and_convert_to_numeric)
                        df['单价_num'] = df['识别单价'].apply(clean_and_convert_to_numeric)
                        df['总价_num'] = df['识别总价'].apply(clean_and_convert_to_numeric)
                        
                        # ✅ 1. 计算基准答案
                        df['计算总价'] = (df['数量_num'] * df['单价_num']).round(2)
                        
                        # ✅ 2. 反向推算数量 (处理除以0的情况)
                        df['[按总价]推算数量'] = np.where(df['单价_num'] != 0, (df['总价_num'] / df['单价_num']).round(2), np.nan)
                        
                        # ✅ 3. 反向推算单价 (处理除以0的情况)
                        df['[按总价]推算单价'] = np.where(df['数量_num'] != 0, (df['总价_num'] / df['数量_num']).round(2), np.nan)
                        
                        # ✅ 4. 生成简单的状态
                        df['状态'] = np.where(np.isclose(df['计算总价'], df['总价_num']), '✅ 一致', '⚠️ 需核对')
                        df.loc[df['计算总价'].isna() | df['总价_num'].isna(), '状态'] = '❔ 信息不足'

                        # --- 核心诊断逻辑结束 ---
                        
                        final_cols = ["品名", "识别数量", "识别单价", "识别总价", "计算总价", "[按总价]推算数量", "[按总价]推算单价", "状态"]
                        st.session_state.results[file_id] = df[final_cols]
                        
                        st.success("✅ 诊断完成！请查看下面的并列分析表。")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 处理失败，发生未知错误：{e}")

            # 在按钮下方直接显示结果表格
            if file_id in st.session_state.results:
                st.dataframe(st.session_state.results[file_id], use_container_width=True)
                st.caption("👆 请直接对比上方表格中的数字，快速定位问题。")


# --- 统一编辑与导出 ---
if st.session_state.results:
    st.divider()
    st.header("📝 统一编辑与导出")

    all_dfs = list(st.session_state.results.values())
    if all_dfs:
        # 在最终编辑和导出时，可以保留所有列，因为它们都有参考价值
        merged_df = pd.concat(all_dfs, ignore_index=True)

        st.info("您可以在下表中修改 **识别数量**、**识别单价**、**识别总价**。其它列仅供参考。")
        edited_df = st.data_editor(
            merged_df,
            num_rows="dynamic",
            use_container_width=True,
            height=300,
            # 锁定所有计算和推算列
            disabled=["计算总价", "[按总价]推算数量", "[按总价]推算单价", "状态"]
        )

        st.subheader("📥 导出为 Excel 文件")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='诊断结果')
            writer.sheets['诊断结果'].autofit()
        
        excel_data = output.getvalue()
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"订单诊断结果_{now}.xlsx"
        
        st.download_button(
            label="✅ 点击下载【诊断后】的Excel",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
