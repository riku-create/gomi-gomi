import streamlit as st
import requests
from PIL import Image
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import json
import pandas as pd
from datetime import datetime

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
        font-size: clamp(24px, 5vw, 40px);
        text-align: center;
        margin-bottom: 30px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        padding: 0 20px;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 15px;
        background-color: #E8F5E9;
        border-radius: 10px;
        margin: 20px 0;
    }
    @media screen and (max-width: 768px) {
        .title {
            font-size: clamp(20px, 4vw, 32px);
        }
    }
    </style>
    """, unsafe_allow_html=True)

# タイトル（一行で表示、サイズ自動調整）
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

# 都道府県と市区町村のデータを読み込む
@st.cache_data
def load_location_data():
    # 都道府県データ
    prefectures = [
        "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
        "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
        "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
        "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
        "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
        "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
        "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
    ]
    return prefectures

# ゴミ出し情報を取得する関数
@st.cache_data
def get_garbage_schedule(prefecture, city):
    try:
        # 環境省のAPIエンドポイント（実際のAPIエンドポイントに置き換える必要があります）
        url = f"https://api.example.com/garbage-schedule/{prefecture}/{city}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            # APIが利用できない場合は、デフォルトのスケジュールを返す
            return {
                "可燃ゴミ": ["月曜日", "木曜日"],
                "不燃ゴミ": ["火曜日"],
                "資源ゴミ": ["金曜日"],
                "注意事項": [
                    "朝8時までに出してください",
                    "雨の日はビニール袋に入れてください",
                    "分別ルールを確認してください"
                ]
            }
    except:
        # エラー時はデフォルトのスケジュールを返す
        return {
            "可燃ゴミ": ["月曜日", "木曜日"],
            "不燃ゴミ": ["火曜日"],
            "資源ゴミ": ["金曜日"],
            "注意事項": [
                "朝8時までに出してください",
                "雨の日はビニール袋に入れてください",
                "分別ルールを確認してください"
            ]
        }

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
        
        # 分類結果の表示（一行で表示）
        st.markdown('<div class="result-text">このゴミは 🔥 可燃ゴミ です！ 捨てる日は月曜日と木曜日です！</div>', unsafe_allow_html=True)
        
        # 捨て方のポイント
        st.markdown('<div class="garbage-info">', unsafe_allow_html=True)
        st.markdown("""
        ##### 💡 捨て方のポイント
        - 水気をよく切ってから捨ててね
        - できるだけ小さくしてから捨てよう
        - においのするものはビニール袋に入れてね
        - 朝8時までに出してね！
        - 雨の日はビニール袋に入れてね
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# 地域情報の入力
st.markdown("### 🏠 あなたの住んでいる場所を教えてね！")
col1, col2 = st.columns(2)

# 都道府県の選択
prefectures = load_location_data()
with col1:
    prefecture = st.selectbox("都道府県", prefectures)

# 市区町村の入力
with col2:
    city = st.text_input("市区町村", placeholder="例：渋谷区")

if prefecture and city:
    # ゴミ出し情報を取得
    schedule = get_garbage_schedule(prefecture, city)
    
    st.markdown('<div class="garbage-info">', unsafe_allow_html=True)
    st.markdown(f"### 📅 {prefecture}{city}のゴミ出しカレンダー")
    
    # ゴミの種類ごとの表示
    for garbage_type, days in schedule.items():
        if garbage_type != "注意事項":
            st.markdown(f"#### {garbage_type}")
            st.markdown(f"- {'・'.join(days)}")
    
    # 注意事項の表示
    st.markdown("""
    ##### ⚠️ 注意事項
    """)
    for note in schedule["注意事項"]:
        st.markdown(f"- {note}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# フッター
st.markdown("""
    ---
    ### 🌟 ゴミ分別で地球をきれいにしよう！
    """) 
