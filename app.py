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
    page_title="Gemini 稳健队列诊断",
    page_icon="🚦",
    layout="wide"
)

# --- 应用标题和说明 ---
st.title("🚦 Gemini 智能订单诊断工具 V4.0 (稳健队列版)")
st.markdown("""
欢迎使用最稳定的队列处理版本！本工具为解决云端资源限制而设计，确保大批量图片也能逐一成功处理。
- **任务队列**：上传的所有图片将进入一个“待处理”队列。
- **逐一处理**：点击“处理下一张”按钮，应用将一次只识别一张图片，有效避免内存溢出。
- **状态清晰**：时刻了解处理进度，还剩多少张待处理。
""")

# --- 会话状态 (Session State) 初始化 ---
# 我们需要更复杂的状态管理来支持队列
if "file_list" not in st.session_state:
    st.session_state.file_list = [] # 存储所有上传的文件对象
if "results" not in st.session_state:
    st.session_state.results = {} # 存储已识别的结果
if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = [] # 存储已处理文件的ID

# --- API 密钥配置 和 模型初始化 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    st.error(f"初始化失败，请检查 API 密钥或模型名称是否正确: {e}")
    st.stop()

# --- Prompt 和数据清洗函数 (保持不变) ---
PROMPT_TEMPLATE = """...""" # 省略，和原来一样
def clean_and_convert_to_numeric(value):
    if value is None or not isinstance(value, str) or value.strip() == "": return np.nan
    numbers = re.findall(r'[\d\.]+', str(value))
    if numbers:
        try: return float(numbers[0])
        except (ValueError, IndexError): return np.nan
    return np.nan

# --- 文件上传与队列管理 ---
st.header("STEP 1: 上传所有订单图片")
uploaded_files = st.file_uploader(
    "请在此处上传一张或多张图片",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    # 更新文件列表，同时保持现有结果
    st.session_state.file_list = uploaded_files

# --- 队列处理控制中心 ---
if st.session_state.file_list:
    files_to_process = [f for f in st.session_state.file_list if f.file_id not in st.session_state.processed_ids]
    
    st.header("STEP 2: 逐一处理识别任务")
    
    if not files_to_process:
        st.success("🎉 所有图片均已处理完毕！请在下方查看、编辑和导出结果。")
    else:
        total_count = len(st.session_state.file_list)
        remaining_count = len(files_to_process)
        st.info(f"**任务队列状态**：共 {total_count} 张图片，还剩 **{remaining_count}** 张待处理。")
        
        # 获取队列中的下一个任务
        next_file_to_process = files_to_process[0]
        
        st.subheader(f"下一个待处理：**{next_file_to_process.name}**")
        st.image(next_file_to_process, width=200, caption="待处理图片预览")
        
        if st.button("👉 处理这一张", use_container_width=True, type="primary"):
            with st.spinner(f"正在识别 {next_file_to_process.name}..."):
                try:
                    file_id = next_file_to_process.file_id
                    image = Image.open(next_file_to_process).convert("RGB")
                    
                    response = model.generate_content([PROMPT_TEMPLATE, image])
                    cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                    data = json.loads(cleaned_text)
                    
                    df = pd.DataFrame.from_records(data)
                    df.rename(columns={"数量": "识别数量", "单价": "识别单价", "总价": "识别总价"}, inplace=True)
                    
                    # ... (诊断逻辑和V3.3完全一样)
                    expected_cols = ["品名", "识别数量", "识别单价", "识别总价"]
                    for col in expected_cols:
                        if col not in df.columns: df[col] = ""
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
                    
                    # 标记此文件为已处理
                    st.session_state.processed_ids.append(file_id)
                    st.success(f"✅ {next_file_to_process.name} 处理成功！")
                    st.rerun()

                except Exception as e:
                    st.error(f"处理 {next_file_to_process.name} 时发生错误: {e}")
                    # 即使失败，也标记为已处理，避免队列卡住
                    st.session_state.processed_ids.append(next_file_to_process.file_id)
                    st.session_state.results[next_file_to_process.file_id] = pd.DataFrame([{"品名": f"识别失败: {e}", "状态": "❌ 错误"}])
                    st.rerun()


# --- 结果展示与导出 ---
st.header("STEP 3: 查看、编辑与导出结果")
if not st.session_state.results:
    st.info("尚未处理任何图片。")
else:
    # 按照上传顺序展示结果
    for file in st.session_state.file_list:
        if file.file_id in st.session_state.results:
            with st.expander(f"📄 订单：{file.name} (已处理)", expanded=False):
                st.dataframe(st.session_state.results[file.file_id], use_container_width=True)

    # --- 统一编辑与导出 ---
    all_dfs = [df for df in st.session_state.results.values() if '识别数量' in df.columns]
    if all_dfs:
        st.subheader("统一编辑区")
        merged_df = pd.concat(all_dfs, ignore_index=True)
        edited_df = st.data_editor(
            merged_df,
            num_rows="dynamic",
            use_container_width=True,
            height=300,
            disabled=["计算总价", "[按总价]推算数量", "[按总价]推算单价", "状态"]
        )

        st.subheader("导出为 Excel 文件")
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
