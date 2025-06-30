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
    page_icon="🍲",
    layout="wide"
)

# --- 应用标题和说明 ---
st.title("🍲 Gemini 智能订单诊断工具 V4.4 (分类与动态计算版)")
st.markdown("""
欢迎使用！本版本在交互式编辑的基础上，新增 **智能分类** 与 **动态计算** 功能！
- **智能分类**：AI 会自动将商品归入 `鱼类`、`蔬菜类`、`肉类` 等类别。
- **动态计算**：您在表格中修改 `数量` 或 `单价`后，`计算总价` 和 `状态` **会立即自动更新**！
- **图表对照**：在每个识别结果旁直接显示原始图片，核对一目了然。
- **数据联动**：您在上方任何表格中所做的修改，都会 **立即自动同步** 到下方的汇总总表及最终的Excel导出文件中。
""")

# --- 会话状态 (Session State) 初始化 ---
if "file_list" not in st.session_state: st.session_state.file_list = []
if "results" not in st.session_state: st.session_state.results = {}
if "processed_ids" not in st.session_state: st.session_state.processed_ids = []
if "processing_active" not in st.session_state: st.session_state.processing_active = False

# --- 安全设置 ---
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- API 密钥配置 和 模型初始化 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest", safety_settings=SAFETY_SETTINGS)
except Exception as e:
    st.error(f"初始化失败，请检查 API 密钥或模型名称是否正确: {e}")
    st.stop()

# --- ✅ [V4.4 核心升级] Prompt ---
PROMPT_TEMPLATE = """
你是一个顶级的、非常严谨的餐饮行业订单数据录入专家。订单内容主要是餐厅后厨采购的食材。
请仔细识别这张手写订单图片，并提取每一行商品的'品名'、'数量'、'单价'和'总价'，并对商品进行分类。

请严格遵守以下规则：
1.  最终必须输出一个格式完美的 JSON 数组。
2.  数组中的每一个 JSON 对象代表一个商品，且必须包含五个键： "品名", "数量", "单价", "总价", "分类"。
3.  新增规则：请为每个商品增加一个 '分类' 键。根据商品名称，将其归入以下类别之一：'鱼类', '猪肉类', '鸡肉类', '鸭肉类', '蔬菜类', '牛肉类', '羊肉类', '调料类', '消耗品类'。如果无法判断，可以设为"其他"。
    - 示例: '黄鱼' 归入 '鱼类'; '五花肉' 归入 '猪肉类'; '香菜' 归入 '蔬菜类'; '洗洁精' 归入 '消耗品类'。
4.  品名和分类必须是文字。数量、单价、总价应该是数字或能解析为数字的字符串。
5.  如果图片中的某一行缺少某个信息（例如没有写单价），请将对应的值设为空字符串 ""。
6.  如果某个文字或数字非常模糊，无法确定，也请设为空字符串 ""。
7.  **你的回答必须是纯粹的、可以直接解析的 JSON 文本**。绝对不要包含任何解释、说明文字、或者 Markdown 的 ```json ``` 标记。
"""

# --- 数据清洗与计算函数 ---
def clean_and_convert_to_numeric(value):
    if value is None or not isinstance(value, str) or value.strip() == "": return np.nan
    numbers = re.findall(r'[\d\.]+', str(value))
    if numbers:
        try: return float(numbers[0])
        except (ValueError, IndexError): return np.nan
    return np.nan

# ✅ [V4.4 核心升级] 新增可重用的计算函数
def recalculate_dataframe(df):
    """对给定的DataFrame进行清洗、计算和状态更新"""
    df_copy = df.copy()

    # 确保关键列存在，防止用户删除列导致报错
    expected_cols = ["品名", "分类", "识别数量", "识别单价", "识别总价"]
    for col in expected_cols:
        if col not in df_copy.columns:
            df_copy[col] = ""

    # 应用清洗和转换
    df_copy['数量_num'] = df_copy['识别数量'].apply(clean_and_convert_to_numeric)
    df_copy['单价_num'] = df_copy['识别单价'].apply(clean_and_convert_to_numeric)
    df_copy['总价_num'] = df_copy['识别总价'].apply(clean_and_convert_to_numeric)

    # 重新计算
    df_copy['计算总价'] = (df_copy['数量_num'] * df_copy['单价_num']).round(2)
    df_copy['[按总价]推算数量'] = np.where(df_copy['单价_num'] != 0, (df_copy['总价_num'] / df_copy['单价_num']).round(2), np.nan)
    df_copy['[按总价]推算单价'] = np.where(df_copy['数量_num'] != 0, (df_copy['总价_num'] / df_copy['数量_num']).round(2), np.nan)

    # 重新判断状态 (equal_nan=True 让两个NaN被视为一致)
    df_copy['状态'] = np.where(np.isclose(df_copy['计算总价'], df_copy['总价_num'], equal_nan=True), '✅ 一致', '⚠️ 需核对')
    df_copy.loc[df_copy['计算总价'].isna() | df_copy['总价_num'].isna(), '状态'] = '❔ 信息不足'

    # 定义最终列顺序，增加了'分类'
    final_cols = ["品名", "分类", "识别数量", "识别单价", "识别总价", "计算总价", "[按总价]推算数量", "[按总价]推算单价", "状态"]
    
    # 确保所有最终列都存在
    for col in final_cols:
        if col not in df_copy.columns:
            df_copy[col] = np.nan
            
    return df_copy[final_cols]


# --- 文件上传与队列管理 ---
st.header("STEP 1: 上传所有订单图片")
uploaded_files = st.file_uploader("请在此处上传一张或多张图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    new_file_ids = {f.file_id for f in uploaded_files}
    old_file_ids = {f.file_id for f in st.session_state.file_list}
    if new_file_ids != old_file_ids:
        st.session_state.file_list = uploaded_files
        st.session_state.results = {}
        st.session_state.processed_ids = []
        st.session_state.processing_active = False
        st.info("检测到新的文件列表，已重置处理队列。")
        st.rerun()

# --- 队列处理控制中心 ---
if st.session_state.file_list:
    files_to_process = [f for f in st.session_state.file_list if f.file_id not in st.session_state.processed_ids]
    total_count = len(st.session_state.file_list)
    remaining_count = len(files_to_process)
    processed_count = total_count - remaining_count

    st.header("STEP 2: 自动处理识别任务")
    st.progress(processed_count / total_count if total_count > 0 else 0, text=f"处理进度：{processed_count} / {total_count}")

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.processing_active and files_to_process:
            if st.button("🚀 开始批量处理", use_container_width=True, type="primary"):
                st.session_state.processing_active = True
                st.rerun()
    with col2:
        if st.session_state.processing_active:
            if st.button("⏹️ 停止处理", use_container_width=True):
                st.session_state.processing_active = False
                st.warning("处理已手动停止。")
                st.rerun()

    if st.session_state.processing_active and files_to_process:
        next_file_to_process = files_to_process[0]
        with st.spinner(f"正在识别 {next_file_to_process.name}... (队列剩余 {remaining_count-1} 张)"):
            try:
                file_id = next_file_to_process.file_id
                image = Image.open(next_file_to_process).convert("RGB")
                response = model.generate_content([PROMPT_TEMPLATE, image])
                if not response.text or not response.text.strip():
                    raise ValueError("模型返回了空内容。可能是图片质量问题或安全策略触发。")

                cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                data = json.loads(cleaned_text)
                df = pd.DataFrame.from_records(data)
                df.rename(columns={"数量": "识别数量", "单价": "识别单价", "总价": "识别总价"}, inplace=True)
                
                # ✅ [V4.4 核心升级] 调用重构的计算函数
                processed_df = recalculate_dataframe(df)

                st.session_state.results[file_id] = processed_df
                st.session_state.processed_ids.append(file_id)
            except Exception as e:
                st.error(f"处理 {next_file_to_process.name} 时出错: {e}")
                st.session_state.processed_ids.append(next_file_to_process.file_id)
                st.session_state.results[next_file_to_process.file_id] = pd.DataFrame([{"品名": f"识别失败", "状态": "❌ 错误"}])
            st.rerun()

    if not files_to_process and total_count > 0:
        if st.session_state.processing_active:
            st.session_state.processing_active = False
            st.success("🎉 所有图片均已自动处理完毕！")
            st.balloons()
            st.rerun()
        else:
            st.info("所有图片均已处理。请在下方查看、编辑和导出结果。")


# --- ✅ [V4.4 核心修改] 结果展示、编辑与动态计算 ---
st.header("STEP 3: 对照图片，编辑结果（编辑后自动重算）")

if not st.session_state.results:
    st.info("尚未处理任何图片。请先上传并开始处理。")
else:
    for file in st.session_state.file_list:
        if file.file_id in st.session_state.results:
            st.markdown("---")
            st.subheader(f"📄 订单：{file.name}")
            
            col_img, col_editor = st.columns([1, 2])

            with col_img:
                st.image(file, caption="原始订单图片", use_column_width=True)

            with col_editor:
                current_df = st.session_state.results[file.file_id]
                
                # 捕获用户编辑后的 DataFrame
                edited_df = st.data_editor(
                    current_df,
                    key=f"editor_{file.file_id}",
                    num_rows="dynamic",
                    use_container_width=True,
                    # 禁用自动计算的列，但允许用户修改 "分类"
                    disabled=["计算总价", "[按总价]推算数量", "[按总价]推算单价", "状态"]
                )

                # ✅ [V4.4 核心升级] 动态计算：对编辑后的数据立即重新计算
                recalculated_edited_df = recalculate_dataframe(edited_df)
                
                # 将完全更新（包含最新计算结果）的 DataFrame 写回 session_state
                st.session_state.results[file.file_id] = recalculated_edited_df

    # --- 汇总预览与导出 ---
    st.markdown("---")
    st.header("STEP 4: 预览汇总结果并导出")

    # 从 session_state 中收集所有（可能已被编辑和重算过的）DataFrame
    all_dfs = [df for df in st.session_state.results.values() if isinstance(df, pd.DataFrame) and '识别数量' in df.columns]
    
    if all_dfs:
        st.subheader("统一结果预览区 (根据您的修改实时更新)")
        
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        st.dataframe(merged_df, use_container_width=True, height=300)
        
        st.subheader("导出为 Excel 文件")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            merged_df.to_excel(writer, index=False, sheet_name='诊断结果')
            worksheet = writer.sheets['诊断结果']
            for i, col in enumerate(merged_df.columns):
                column_len = max(merged_df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)

        excel_data = output.getvalue()
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"订单诊断结果_{now}.xlsx"
        
        st.download_button(
            label="✅ 点击下载【最终修正后】的Excel",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
