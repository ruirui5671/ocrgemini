# V3.4 - Diagnostic Mode
import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io
import json
import datetime
import re
import numpy as np

# --- DIAGNOSTIC PRINT ---
print(">>> Running app.py V3.4 - Diagnostic Mode <<<")

# --- 页面基础配置 ---
st.set_page_config(
    page_title="Gemini 批量诊断",
    page_icon="🚀",
    layout="wide"
)

# --- 应用标题和说明 ---
st.title("🚀 Gemini 智能订单诊断工具 V3.4 (诊断模式)")
st不要上传手机拍的照片。请在您的电脑上，用截图工具**截取一小块屏幕**（比如就截取您桌面的一部分），保存成一张图片。这种截图文件通常非常小，只有几十或几百 KB。

然后，**只上传这一张非常小的截图文件**，点击识别按钮，再去观察日志。

*   **如果这次识别成功了**，那么我们就 100% 确定了问题就是图片太大导致的内存溢出。
*   **如果这次还是不行**，那么问题就更深层，我们需要执行第二步。

#### **第二步：植入“侦察兵”的诊断代码**

如果小图片测试也失败了，我们就需要在代码里放上很多“侦察兵”（`print` 语句），来报告它到底死在了哪一步。

请您**用下面的诊断版代码，完整替换您的 `app.py`**。这个版本不会解决问题，但**它一定会在日志里留下线索**。

```python
# V3.4 - 深度诊断版
import streamlit as st
import google.generativeai as genai
from PIL import Image
.markdown("当前处于诊断模式，请关注后台日志。")

# --- API 密钥配置 和 模型初始化 ---
# --- DIAGNOSTIC PRINT ---
print(">>> Preparing to configure Gemini API...")
try:
    # --- DIAGNOSTIC PRINT ---
    print(">>> Attempting to read API Key from st.secrets...")
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # --- DIAGNOSTIC PRINT ---
    print(">>> Successfully read API Key from secrets! Key starts with: " + api_key[:6])
    
    genai.configure(api_key=api_key)
    # --- DIAGNOSTIC PRINT ---
    print(">>> genai.configure() executed successfully.")
    
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    # --- DIAGNOSTIC PRINT ---
    print(">>> GenerativeModel initialized successfully.")
    st.success("API 初始化成功！可以开始上传图片了。")

except Exception as e:
    # --- DIAGNOSTIC PRINT ---
    print(f"!!! CAUGHT EXCEPTION DURING INITIALIZATION: {eimport pandas as pd
import io
import json
import datetime
import re
import numpy as np

# --- 侦察兵 #1：确认脚本开始运行 ---
print("--- [DEBUG] SCRIPT STARTED ---")

# --- 页面基础配置 ---
st.set_page_config(
    page_title="Gemini 批量诊断",
    page_icon="🚀",
    layout="wide"
)
st.title("🚀 Gemini 智能订单诊断工具 V3.4 (深度诊断版)")
st.markdown("欢迎使用！这是一个用于深度调试的版本。")

# --- API 密钥配置 和 模型初始化 ---
try:
    # --- 侦察兵 #2：确认开始配置API ---
    print("--- [DEBUG] CONFIGURING API KEY... ---")
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    # --- 侦察兵 #3：确认API配置成功 ---
    print("--- [DEBUG] API KEY CONFIGURED SUCCESSFULLY. ---")
except Exception as e:
    # --- 侦察兵 #4：捕}")
    st.error(f"API 初始化失败，后台日志已记录错误。请将日志中的 '!!! CAUGHT EXCEPTION' 信息提供给支持人员。错误详情: {e}")
    st.stop()


# --- Prompt 保持不变 ---
PROMPT_TEMPLATE = """
你是一个顶级的、非常严谨的订单数据录入专家。
请仔细识别这张手写订单图片，并提取每一行商品的'品名'、'数量'、'单价'和'总价'。
请严格遵守以下规则：
1.  最终必须输出一个格式完美的 JSON 数组。
2.  数组中的每一个 JSON 对象代表一个商品，且必须包含四个键： "品名", "数量", "单价", "总价"。
3.  如果图片获API配置错误 ---
    print(f"--- [FATAL DEBUG] API CONFIGURATION FAILED: {e} ---")
    st.error(f"初始化失败，请检查 API 密钥或模型名称是否正确: {e}")
    st.stop()

# ... (PROMPT 和函数定义部分省略，它们和原来一样) ...
PROMPT_TEMPLATE = "..." # 省略，和原来一样
def clean_and_convert_to_numeric(value): # 省略，和原来一样
    # ...
    return np.nan

# --- 会话状态 (Session State) 初始化 ---
if "results" not in st.session_state:
    st中的某一行缺少某个信息（例如没有写单价），请将对应的值设为空字符串 ""。
4.  如果某个文字或数字非常模糊，无法确定，也请设为空字符串 ""。
5.  **你的回答必须是纯粹的、可以直接解析的 JSON 文本**。绝对不要包含任何解释、说明文字、或者 Markdown 的 ```json ``` 标记。
"""

# --- 会话状态 (Session State) 初始化 ---
if "results" not in st.session_state:
    st.session_state.results = {}

.session_state.results = {}

# --- 文件上传组件 ---
files = st.file_uploader(
    "📤 STEP 1: 上传所有订单图片",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if files:
    if st.button("🚀 STEP 2: 一次性识别所有图片", use_container_width=True, type="primary"):
        # --- # --- 数据清洗函数 ---
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
files = st.file_uploader(侦察兵 #5：确认按钮被点击 ---
        print("--- [DEBUG] IDENTIFY BUTTON CLICKED. ---")
        
        files_to_process = [f for f in files if f.file_id not in st.session_state.results]
        
        if not files_to_process:
            st.toast("所有已上传的图片都识别过啦！")
        else:
            total_files
    "📤 STEP 1: 上传所有订单图片",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# --- 批量处理逻辑 ---
if files:
    if st.button("🚀 STEP 2: 一次性识别所有图片", use_container_width=True, type="primary"):
        # ...后续代码和之前V3.3版本一致，为了简洁此处省略...
         = len(files_to_process)
            progress_bar = st.progress(0, text="准备开始批量识别...")

            for i, file in enumerate(files_to_process):
                file_id = file.file_id
                progress_text = f"正在识别第 {i + 1}/{total_files} 张图片: {file.name}"
                progress_bar.progress((i + 1) / total_files, text=progress_text)
                
                # --- 侦察兵 #6：# ...如果初始化成功，这部分逻辑不会有问题...
        files_to_process = [f for f in files if f.file_id not in st.session_state.results]
        
        if not files_to_process:
            st.toast("所有已上传的图片都识别过啦！")
        else:
            total_files = len(files_to_process)
            progress_bar = st.progress报告正在处理哪个文件 ---
                print(f"--- [DEBUG] PROCESSING FILE: {file.name} (ID: {file_id}) ---")
                
                try:
                    # --- 侦察兵 #7：报告准备打开图片 ---
                    print(f"--- [DEBUG] STEP A: Opening image file... ---")
                    image = Image.open(file).convert("RGB")
                    # --- 侦察兵 #8：报告(0, text="准备开始批量识别...")

            for i, file in enumerate(files_to_process):
                file_id = file.file_id
                progress_text = f"正在识别第 {i + 1}/{total_files} 张图片: {file.name}"
                progress_bar.progress((图片打开成功，准备调用API ---
                    print(f"--- [DEBUG] STEP B: Image opened successfully. Calling Gemini API... ---")
                    response = model.generate_content([PROMPT_TEMPLATE, image])
                    # --- 侦察兵 #9：报告API调用成功 ---
                    print(f"--- [DEBUG] STEP C:i + 1) / total_files, text=progress_text)
                
                try:
                    image = Image.open(file).convert("RGB")
                    response = model.generate_content([PROMPT Gemini API call successful. Processing response... ---")
                    
                    cleaned_text = response.text.strip().removeprefix("```json").removesuffix("_TEMPLATE, image])
                    cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                    data = json.loads(cleaned_text)
                    
                    df = pd.DataFrame.from_records(data)
                    df.rename(columns={"数量": "识别数量```").strip()
                    data = json.loads(cleaned_text)
                    df = pd.DataFrame.from_records(data)
                    # ... (后续处理代码省略) ...
                    
                    st.session_state.results[file_id] = df # 简化", "单价": "识别单价", "总价": "识别总价"}, inplace=True)
                    
                    expected_cols = ["品名", "识别数量", "识别单价", "识别总价"]
                    for col in expected_cols:
                        if col not in df.columns:
                            df[col] = ""
                    存储，只为调试

                except Exception as e:
                    # --- 侦察兵 #10：报告捕获到明确的错误 ---
                    print(f"--- [FATAL DEBUG] EXCEPTION CAUGHT FOR
                    df['数量_num'] = df['识别数量'].apply(clean_and_convert_to_numeric)
                    df['单价_num'] = df['识别单价'].apply(clean_and_convert_to_numeric)
                    df['总价_num'] = df['识别总价'].apply(clean_and_convert_to_numeric)
                    
                    df['计算总价'] = (df[' FILE {file.name}: {e} ---")
                    st.session_state.results[file_id] = pd.DataFrame([{"品名": f"识别失败: {e}", "状态": "❌ 错误"}])

            progress_bar.empty()
            st.success("🎉 批量处理循环结束！")数量_num'] * df['单价_num']).round(2)
                    df['[按总价]推算数量'] = np.where(df['单价_num'] != 0, (df['
            st.rerun()

    st.subheader("STEP 3: 逐一核对诊断结果")
    # ... (后续展示代码省略) ...
```总价_num'] / df['单价_num']).round(2), np.nan)
                    df['[按总价]推算单价'] = np.where(df['数量_num'] != 0, (df['总价_num'] / df['数量_num']).round(2), np.nan
**(请注意，我省略了部分未改动的代码以缩短篇幅，您需要将这些 print 语句整合到您完整的 V3.3 代码中)**)
                    df['状态'] = np.where(np.isclose(df['计算总价'], df['总价_num']), '✅ 一致', '⚠️ 需核对')
                    df.loc[df['计算总价'].isna() | df['总价_num'].isna(), '状态'] = '❔

**在您用这个新代码替换、推送并重启应用后：**
1.  再次上传一张**小图片**并点击识别。
2.  现在，**日志里一定会留下新的线索！** 信息不足'

                    final_cols = ["品名", "识别数量", "识别单价", "识别总价", "计算总价", "[按总价]推算数量", "[按总价]推算单价",
3.  请观察日志，看看最后出现的是哪个 `[DEBUG]` 信息。
    *   如果 "状态"]
                    st.session_state.results[file_id] = df[final_cols]

                except Exception as e:
                    st.session_state.results[file_id] = pd.最后一条是 `STEP A: Opening image file...`，然后就没了，那100%是内存溢DataFrame([{"品名": f"识别失败: {e}", "状态": "❌ 错误"}])

出。
    *   如果最后一条是 `STEP B: ... Calling Gemini API...`，那说明是            progress_bar.empty()
            st.success("🎉 所有新图片识别完成！")
            st.rer调用 API 的网络环节卡住了。
    *   如果日志里出现了 `[FATAL DEBUG]`，那就更un()


    # --- 独立展示每张图片及其结果 ---
    st.subheader("STEP 3: 逐一核对诊断结果")
    for file in files:
        file_id = file.file_好了，直接告诉了我们错误原因。

**请您先进行“小图片测试”，然后告诉我结果。** 这一步是我们解决问题的最快路径！
