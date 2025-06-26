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
st.title("🧠 Gemini OCR: 智能手写订单识别工具 V2")
st.markdown("""
欢迎使用 V2 版本！此版本已升级为**自动结构化识别**模式。
- **自动提取字段**：上传手写订单图片，Gemini 将尝试自动识别出 `品名`、`数量`、`单价` 等关键信息。
- **统一编辑和导出**：所有识别结果会合并在一张表格中，您可以方便地进行修改、补充，并一键导出为 Excel 文件。
""")

# --- API 密钥配置 和 模型初始化 ---
# 从 Streamlit Secrets 加载 Google API Key
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest") # 建议使用最新的模型以获得更好的JSON输出效果
except Exception as e:
    st.error(f"API密钥配置错误，请检查.streamlit/secrets.toml文件: {e}")
    st.stop()


# --- Gemini 的指令 (Prompt) ---
# 这是实现结构化识别的核心
PROMPT_TEMPLATE = """
你是一个专业的订单数据录入员。
请仔细识别这张手写进货单图片，并提取每一行的'品名'、'数量'和'单价'。

请严格按照以下要求操作：
1. 将结果整理成一个 JSON 数组格式。
2. 数组中的每个对象代表一个商品，必须包含三个键： "品名", "数量", "单价"。
3. 如果图片中的某一行缺少某个信息（例如没有写单价），请将对应的值留空字符串 ""。
4. 如果某个值无法清晰识别，请尽力猜测或也留空字符串。
5. 最终的输出结果 **只能是 JSON 格式的文本**，不要包含任何解释、说明、或者 markdown 的 ```json ``` 标记。

例如，对于一张包含 "雪花纯生 5箱 85元" 和 "青岛原浆 3箱 120元" 的图片，你应该返回：
[
    { "品名": "雪花纯生", "数量": "5", "单价": "85" },
    { "品名": "青岛原浆", "数量": "3", "单价": "120" }
]
"""

# --- 会话状态 (Session State) 初始化 ---
# 使用字典来存储结果，key为图片索引，value为DataFrame，防止重复处理
if "results" not in st.session_state:
    st.session_state.results = {}

# --- 文件上传组件 ---
files = st.file_uploader(
    "📤 上传一张或多张订单图片 (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if files:
    # --- 遍历处理每张上传的图片 ---
    for i, file in enumerate(files):
        # 使用 Expander 折叠每一张图片的处理区域
        with st.expander(f"📷 第 {i+1} 张图片：{file.name}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("原始图片")
                image = Image.open(file).convert("RGB")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("识别与处理")
                # 为每张图片创建一个唯一的识别按钮
                if st.button(f"🚀 结构化识别第 {i+1} 张", key=f"btn_{i}"):
                    with st.spinner("🧠 Gemini 正在进行结构化识别..."):
                        try:
                            # 将图片转换为字节流
                            buf = io.BytesIO()
                            image.save(buf, format="JPEG")
                            img_bytes = buf.getvalue()

                            # 调用 Gemini API
                            response = model.generate_content([
                                PROMPT_TEMPLATE, # 关键：使用新的结构化指令
                                {"mime_type": "image/jpeg", "data": img_bytes}
                            ])

                            # --- 解析 Gemini 返回的 JSON ---
                            # 移除可能的 markdown 标记
                            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
                            
                            # 解析JSON字符串为Python对象 (list of dicts)
                            data = json.loads(cleaned_text)

                            # 将Python对象转换为Pandas DataFrame
                            df = pd.DataFrame.from_records(data)

                            # --- 保证DataFrame包含所有预期的列 ---
                            expected_cols = ["品名", "数量", "单价"]
                            for col in expected_cols:
                                if col not in df.columns:
                                    df[col] = "" # 如果缺少某列，则添加空列
                            
                            # 调整列顺序并保存到会话状态
                            st.session_state.results[i] = df[expected_cols]
                            st.success("✅ 识别完成！结果已添加到下方总表中。")

                        except json.JSONDecodeError:
                            st.error("❌ 结构化识别失败：模型返回的不是有效的JSON格式。")
                            st.info("Gemini 返回的原始文本：")
                            st.text(response.text if 'response' in locals() else "无返回内容")
                        except Exception as e:
                            st.error(f"❌ 处理失败，发生未知错误：{e}")

                # 如果这张图片已经识别过了，就显示提示信息
                if i in st.session_state.results:
                    st.info("这张图片的结果已在下方表格中。如需重新识别，请再次点击上方按钮。")


# --- 合并、编辑和导出 ---
if st.session_state.results:
    st.divider()
    st.header("📝 统一编辑与导出")

    # 从会话状态中获取所有已识别的 DataFrame
    all_dfs = list(st.session_state.results.values())

    if all_dfs:
        # 使用concat合并所有DataFrame
        merged_df = pd.concat(all_dfs, ignore_index=True)

        st.info("您可以在下表中直接修改或添加行。所有修改将一并导出。")
        # 使用 data_editor 提供一个可编辑的表格
        edited_df = st.data_editor(
            merged_df,
            num_rows="dynamic", # 允许用户动态增加或删除行
            use_container_width=True,
            height=300 # 设置一个默认高度
        )

        # --- 导出为 Excel ---
        st.subheader("📥 导出Excel文件")
        if st.button("生成并下载 Excel 文件"):
            # 使用BytesIO作为内存中的文件缓冲区
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='识别结果')
                # 可以添加自动调整列宽等操作
                writer.sheets['识别结果'].autofit()

            # 获取缓冲区的二进制数据
            excel_data = output.getvalue()

            # 生成带时间戳的文件名
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"订单识别结果_{now}.xlsx"
            
            # 提供下载按钮
            st.download_button(
                label="✅ 点击这里下载 Excel",
                data=excel_data,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("Excel 文件已生成！")
