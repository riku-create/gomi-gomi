import streamlit as st
import requests
from PIL import Image
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# ページ設定
st.set_page_config(
    page_title="ゴミ分別アシスタント",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        font-size: 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .garbage-info {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .title {
        color: #2E7D32;
        font-size: 40px;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# タイトル
st.markdown('<h1 class="title">♻️ ゴミ分別アシスタント ♻️</h1>', unsafe_allow_html=True)

# 説明文
st.markdown("""
    ### 👋 こんにちは！ゴミ分別を手伝うよ！
    ゴミの写真を撮って、どこに捨てればいいか教えてあげるね！
    """)

# モデルの読み込み
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    return processor, model

processor, model = load_model()

# 画像アップロード
st.markdown("### 📸 ゴミの写真を撮ってね！")
uploaded_file = st.file_uploader("ここに写真をドラッグ＆ドロップするか、クリックして選んでね！", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption='📸 アップロードされた写真', use_column_width=True)
    
    # 画像の分類
    with st.spinner('🔍 ゴミの種類を調べているよ...'):
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # 分類結果の表示
        st.markdown('<div class="garbage-info">', unsafe_allow_html=True)
        st.markdown("### 🎯 分別結果")
        st.markdown("#### このゴミは...")
        st.markdown("##### 🔥 可燃ゴミ")
        st.markdown("""
        ##### 💡 捨て方のポイント
        - 水気をよく切ってから捨ててね
        - できるだけ小さくしてから捨てよう
        - においのするものはビニール袋に入れてね
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# 地域情報の入力
st.markdown("### 🏠 あなたの住んでいる場所を教えてね！")
col1, col2 = st.columns(2)
with col1:
    prefecture = st.text_input("都道府県", placeholder="例：東京都")
with col2:
    city = st.text_input("市区町村", placeholder="例：渋谷区")

if prefecture and city:
    st.markdown('<div class="garbage-info">', unsafe_allow_html=True)
    st.markdown(f"### 📅 {prefecture}{city}のゴミ出しカレンダー")
    
    # ゴミの種類ごとの表示
    garbage_types = {
        "🔥 可燃ゴミ": "毎週月曜日・木曜日",
        "💎 不燃ゴミ": "毎週火曜日",
        "♻️ 資源ゴミ": "毎週金曜日"
    }
    
    for garbage_type, schedule in garbage_types.items():
        st.markdown(f"#### {garbage_type}")
        st.markdown(f"- {schedule}")
    
    st.markdown("""
    ##### ⏰ ゴミ出しの時間
    - 朝8時までに出してね！
    - 雨の日はビニール袋に入れてね
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# フッター
st.markdown("""
    ---
    ### 🌟 ゴミ分別で地球をきれいにしよう！
    """) 
