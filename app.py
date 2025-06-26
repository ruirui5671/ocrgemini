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
    page_title="Gemini 批量诊断",
    page_icon="🚀",
    layout="wide"
)

# --- 应用标题和说明 ---
st.title("🚀 Gemini 智能订单诊断工具 V3.3 (批量处理版)")
st.markdown("""
欢迎使用全新优化的批量处理版本！现在，您可以一次性完成所有订单的诊断。
- **批量识别**：上传所有图片后，只需 **点击一次按钮**，即可识别全部图片。
- **优化布局**：缩小了图片预览尺寸，让数据表格更突出，界面更清爽。
- **独立展示**：每张图片及其诊断结果会独立展示，方便您逐一核对。
""")

# --- API 密钥配置 和 模型初始化 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    st.error(f"初始化失败，请检查 API 密钥或模型名称是否正确: {e}")
    st.stop()


# --- Prompt 保持不变 ---
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
    "📤 STEP 1: 上传所有订单图片",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# --- 批量处理逻辑 ---
if files:
    if st.button("🚀 STEP 2: 一次性识别所有图片", use_container_width=True, type="primary"):
        files_to_process = [f for f in files if f.file_id not in st.session_state.results]
        
        if not files_to_process:
            st.toast("所有已上传的图片都识别过啦！")
        else:
            total_files = len(files_to_process)
            progress_bar = st.progress(0, text="准备开始批量识别...")

            for i, file in enumerate(files_to_process):
                file_id = file.file_id
                progress_text = f"正在识别第 {i + 1}/{total_files} 张图片: {file.name}"
                progress_bar.progress((i + 1) / total_files, text=progress_text)
                
                try:
                    image = Image.open(file).convert("RGB")
                    response = model.generate_content([PROMPT_TEMPLATE, image])
                    cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                    data = json.loads(cleaned_text)
                    
                    df = pd.DataFrame.from_records(data)
                    df.rename(columns={"数量": "识别数量", "单价": "识别单价", "总价": "识别总价"}, inplace=True)
                    
                    expected_cols = ["品名", "识别数量", "识别单价", "识别总价"]
                    for col in expected_cols:
                        if col not in df.columns:
                            df[col] = ""
                    
                    df['数量_num'] = df['识别数量'].apply(clean_and_convert_to_numeric)
                    df['单价_num'] = df['识别单价'].apply(clean_and_convert_to_numeric)
                    df['总价_num'] = df['识别总价'].apply(clean_and_convert_to_numeric)
                    
                    df['计算总价'] = (df['数量_num'] * df['单价_num']).round(2)
                    df['[按总价]推算数量'] = np.where(df['单价_num'] != 0, (df['总价_num'] / df['单价_num']).round(2), np.nan)
                    df['[按总价]推算单价'] = np.where(df['数量_num'] != 0, (df['总价_num'] / df['数量_num']).round(2), np.nan)
                    df['状态'] = np.where(np.isclose(df['计算总价'], df['总价_num']), '✅ 一致', '⚠️ 需核对')
                    df.loc[df['计算总价'].isna() | df['总价_num'].isna(), '状态'] = '❔ 信息不足'

                    final_cols = ["品名", "识别数量", "识别单价", "识别总价", "计算总价", "[按总价]推算数量", "[按总价]推算单价", "状态"]
                    st.session_state.results[file_id] = df[final_cols]

                except Exception as e:
                    st.session_state.results[file_id] = pd.DataFrame([{"品名": f"识别失败: {e}", "状态": "❌ 错误"}])

            progress_bar.empty()
            st.success("🎉 所有新图片识别完成！")
            st.rerun()


    # --- 独立展示每张图片及其结果 ---
    st.subheader("STEP 3: 逐一核对诊断结果")
    for file in files:
        file_id = file.file_id
        with st.expander(f"📄 订单：{file.name}", expanded=True):
            col1, col2 = st.columns([0.5, 1.5])
            
            with col1:
                 # ✅ --- 核心修正点：使用新的参数 use_container_width ---
                st.image(Image.open(file).convert("RGB"), use_container_width=True, caption="订单原图")

            with col2:
                if file_id in st.session_state.results:
                    st.dataframe(st.session_state.results[file_id], use_container_width=True)
                else:
                    st.info("这张图片等待识别...")


# --- 统一编辑与导出 ---
if st.session_state.results:
    st.divider()
    st.header("STEP 4: 统一编辑与导出")

    all_dfs = [df for df in st.session_state.results.values() if '识别数量' in df.columns]
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)

        st.info("您可以在下表中修改 **识别数量**、**识别单价**、**识别总价**。其它列仅供参考。")
        edited_df = st.data_editor(
            merged_df,
            num_rows="dynamic",
            use_container_width=True,
            height=300,
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
