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
st.title("🚦 Gemini 智能订单诊断工具 V4.1 (安全策略优化版)")
st.markdown("""
欢迎使用！本版本已优化 **内容安全策略** 并增强了 **错误处理**，以解决模型可能返回空内容的问题。
- **任务队列**：上传的所有图片将进入一个“待处理”队列。
- **逐一处理**：点击“处理下一张”按钮，应用将一次只识别一张图片，有效避免内存溢出。
- **状态清晰**：时刻了解处理进度，还剩多少张待处理。
""")

# --- 会话状态 (Session State) 初始化 ---
if "file_list" not in st.session_state: st.session_state.file_list = []
if "results" not in st.session_state: st.session_state.results = {}
if "processed_ids" not in st.session_state: st.session_state.processed_ids = []

# --- ✅ 核心修改 1：配置安全设置 ---
# 将安全设置的阈值调整为“全部屏蔽”，这会放宽内容审查，减少因误判导致的空返回
# 注意：这并不意味着不安全，只是降低了模型“过于敏感”的概率
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- API 密钥配置 和 模型初始化 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # 在初始化模型时，传入我们定义好的安全设置
    model = genai.GenerativeModel("gemini-1.5-pro-latest", safety_settings=SAFETY_SETTINGS)
except Exception as e:
    st.error(f"初始化失败，请检查 API 密钥或模型名称是否正确: {e}")
    st.stop()

# --- Prompt (保持不变，但为了完整性粘贴在这里) ---
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
def clean_and_convert_to_numeric(value):
    if value is None or not isinstance(value, str) or value.strip() == "": return np.nan
    numbers = re.findall(r'[\d\.]+', str(value))
    if numbers:
        try: return float(numbers[0])
        except (ValueError, IndexError): return np.nan
    return np.nan

# --- 文件上传与队列管理 ---
st.header("STEP 1: 上传所有订单图片")
uploaded_files = st.file_uploader("请在此处上传一张或多张图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files: st.session_state.file_list = uploaded_files

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
        
        next_file_to_process = files_to_process[0]
        st.subheader(f"下一个待处理：**{next_file_to_process.name}**")
        st.image(next_file_to_process, width=200, caption="待处理图片预览")
        
        if st.button("👉 处理这一张", use_container_width=True, type="primary"):
            with st.spinner(f"正在识别 {next_file_to_process.name}..."):
                try:
                    file_id = next_file_to_process.file_id
                    image = Image.open(next_file_to_process).convert("RGB")
                    
                    response = model.generate_content([PROMPT_TEMPLATE, image])
                    
                    # --- ✅ 核心修改 2：增强的错误处理 ---
                    # 在解析前，先检查返回的文本是否为空
                    if not response.text or not response.text.strip():
                        # 如果是空的，抛出一个更友好的、自定义的错误
                        raise ValueError("模型返回了空内容。这可能是因为图片质量问题或内容安全策略被触发。请尝试另一张图片。")

                    cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                    data = json.loads(cleaned_text)
                    
                    df = pd.DataFrame.from_records(data)
                    df.rename(columns={"数量": "识别数量", "单价": "识别单价", "总价": "识别总价"}, inplace=True)
                    
                    # ... (诊断逻辑和原来一样) ...
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
                    
                    st.session_state.processed_ids.append(file_id)
                    st.success(f"✅ {next_file_to_process.name} 处理成功！")
                    st.rerun()

                except Exception as e:
                    # 现在，这里的e会包含我们自定义的ValueError信息，或者原始的JSONDecodeError
                    st.error(f"处理 {next_file_to_process.name} 时发生错误: {e}")
                    st.session_state.processed_ids.append(next_file_to_process.file_id)
                    st.session_state.results[next_file_to_process.file_id] = pd.DataFrame([{"品名": f"识别失败", "状态": "❌ 错误"}])
                    st.rerun()

# --- 结果展示与导出 (保持不变) ---
st.header("STEP 3: 查看、编辑与导出结果")
# ... (后续代码和V4.0一样)
if not st.session_state.results:
    st.info("尚未处理任何图片。")
else:
    for file in st.session_state.file_list:
        if file.file_id in st.session_state.results:
            with st.expander(f"📄 订单：{file.name} (已处理)", expanded=False):
                st.dataframe(st.session_state.results[file.file_id], use_container_width=True)

    all_dfs = [df for df in st.session_state.results.values() if '识别数量' in df.columns]
    if all_dfs:
        st.subheader("统一编辑区")
        merged_df = pd.concat(all_dfs, ignore_index=True)
        edited_df = st.data_editor(merged_df,num_rows="dynamic",use_container_width=True,height=300,disabled=["计算总价", "[按总价]推算数量", "[按总价]推算单价", "状态"])
        st.subheader("导出为 Excel 文件")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='诊断结果')
            writer.sheets['诊断结果'].autofit()
        excel_data = output.getvalue()
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"订单诊断结果_{now}.xlsx"
        st.download_button(label="✅ 点击下载【诊断后】的Excel",data=excel_data,file_name=file_name,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
