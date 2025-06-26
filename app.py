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
    page_title="Gemini 智能订单诊断",
    page_icon="🕵️",
    layout="wide"
)

# --- 应用标题和说明 ---
st.title("🕵️ Gemini 智能订单诊断工具 V3.0")
st.markdown("""
欢迎使用具备 **根源分析能力** 的全新版本！本工具旨在帮您快速定位订单中的潜在错误。
- **忠实识别**：完整展示识别出的 `数量`、`单价` 和 `总价`。
- **计算对比**：独立计算 `识别数量 × 识别单价` 的结果，供您直接对比。
- **智能诊断**：当计算结果与识别总价不符时，**反向推算出可能的正确数值**，帮您快速定位笔误或识别错误。
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
            col1, col2 = st.columns([0.8, 1.2]) # 让右边表格宽一点

            with col1:
                st.subheader("原始图片")
                image = Image.open(file).convert("RGB")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("识别与诊断分析")
                if st.button(f"🚀 开始智能诊断", key=f"btn_{file_id}"):
                    with st.spinner("🕵️ Gemini 正在进行识别和深度诊断..."):
                        try:
                            response = model.generate_content([PROMPT_TEMPLATE, image])
                            cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                            data = json.loads(cleaned_text)
                            
                            # ✅ 为了清晰，明确重命名列
                            df = pd.DataFrame.from_records(data)
                            df.rename(columns={
                                "数量": "识别数量",
                                "单价": "识别单价",
                                "总价": "识别总价"
                            }, inplace=True)
                            
                            # --- ✅ 核心诊断逻辑开始 ---
                            # 1. 确保所有需要的列都存在
                            expected_cols = ["品名", "识别数量", "识别单价", "识别总价"]
                            for col in expected_cols:
                                if col not in df.columns:
                                    df[col] = ""
                            
                            # 2. 清洗所有识别出的数据为数值
                            df['数量_num'] = df['识别数量'].apply(clean_and_convert_to_numeric)
                            df['单价_num'] = df['识别单价'].apply(clean_and_convert_to_numeric)
                            df['总价_num'] = df['识别总价'].apply(clean_and_convert_to_numeric)
                            
                            # 3. 计算“标准答案”总价
                            df['计算总价'] = (df['数量_num'] * df['单价_num']).round(2)
                            
                            # 4. 【关键一步】反向推算，进行诊断
                            def diagnose_discrepancy(row):
                                calc_total = row['计算总价']
                                rec_total = row['总价_num']
                                
                                # 如果信息不全，无法诊断
                                if pd.isna(calc_total) or pd.isna(rec_total):
                                    return "❔ 信息不足"
                                
                                # 如果完全一致
                                if np.isclose(calc_total, rec_total):
                                    return "✅ 完全一致"
                                
                                # 如果不一致，开始诊断
                                suggestion = f"⚠️ 不一致 (差额: {rec_total - calc_total:.2f})"
                                suggestions = []
                                
                                # 诊断1: 总价不变，数量可能是多少？
                                if row['单价_num'] != 0 and pd.notna(row['单价_num']):
                                    implied_qty = rec_total / row['单价_num']
                                    suggestions.append(f"数量应为 **{implied_qty:.2f}**")
                                
                                # 诊断2: 总价不变，单价可能是多少？
                                if row['数量_num'] != 0 and pd.notna(row['数量_num']):
                                    implied_price = rec_total / row['数量_num']
                                    suggestions.append(f"单价应为 **{implied_price:.2f}**")
                                
                                if suggestions:
                                    suggestion += f"\n可能原因: {' 或 '.join(suggestions)}"
                                    
                                return suggestion

                            df['差异诊断'] = df.apply(diagnose_discrepancy, axis=1)

                            # --- 核心诊断逻辑结束 ---
                            
                            final_cols = ["品名", "识别数量", "识别单价", "识别总价", "计算总价", "差异诊断"]
                            st.session_state.results[file_id] = df[final_cols]
                            
                            st.success("✅ 诊断完成！请查看分析结果。")
                            st.rerun()

                        except json.JSONDecodeError:
                            st.error("❌ 结构化识别失败：模型返回的不是有效的JSON格式。")
                            st.info("模型返回的原始文本：")
                            st.text_area("原始输出", cleaned_text if 'cleaned_text' in locals() else response.text, height=150)
                        except Exception as e:
                            st.error(f"❌ 处理失败，发生未知错误：{e}")

                if file_id in st.session_state.results:
                    st.dataframe(st.session_state.results[file_id], use_container_width=True)
                    st.caption("上方为诊断结果。")

if st.session_state.results:
    st.divider()
    st.header("📝 统一编辑与导出")

    all_dfs = list(st.session_state.results.values())
    if all_dfs:
        # 为了避免干扰，导出时不包含诊断列，只导出干净的数据
        export_cols = ["品名", "识别数量", "识别单价", "识别总价", "计算总价"]
        merged_df = pd.concat(all_dfs, ignore_index=True)[export_cols]

        st.info("您可以在下表中直接修改。建议参考上方的“差异诊断”来修正“识别数量”或“识别单价”。")
        edited_df = st.data_editor(
            merged_df,
            num_rows="dynamic",
            use_container_width=True,
            height=300
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
