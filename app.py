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
    body {
        background: linear-gradient(135deg, #e0f7fa 0%, #fffde7 100%);
    }
    .main {
        background-color: transparent;
    }
    .app-card {
        background: #fff;
        border-radius: 24px;
        box-shadow: 0 6px 24px rgba(44, 62, 80, 0.10);
        padding: 28px 18px 24px 18px;
        margin: 18px 0 28px 0;
        max-width: 480px;
        margin-left: auto;
        margin-right: auto;
    }
    .title {
        color: #2E7D32;
        font-size: clamp(4px, 0.6vw, 7px);
        font-weight: 700;
        text-align: center;
        margin-bottom: 18px;
        white-space: nowrap;
        overflow: visible;
        text-overflow: clip;
        padding: 0 10px;
        width: 100%;
        display: inline-block;
        letter-spacing: 0.05em;
    }
    .stButton>button, .stFileUploader>div>button {
        background: linear-gradient(90deg, #4CAF50 60%, #81C784 100%);
        color: #fff;
        border-radius: 18px;
        padding: 12px 32px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.10);
        border: none;
        margin: 8px 0;
        transition: background 0.2s;
    }
    .stButton>button:hover, .stFileUploader>div>button:hover {
        background: linear-gradient(90deg, #388E3C 60%, #66BB6A 100%);
    }
    .stTextInput>div>input, .stSelectbox>div>div>div>input {
        border-radius: 14px;
        border: 1.5px solid #B2DFDB;
        padding: 12px 16px;
        font-size: 16px;
        background: #f9fbe7;
        margin-bottom: 10px;
    }
    .result-text {
        font-size: 22px;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 15px;
        background-color: #E8F5E9;
        border-radius: 14px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.10);
    }
    .garbage-info {
        background-color: #f9fbe7;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.08);
        margin: 10px 0;
    }
    .item-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 16px;
    }
    .item-card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.10);
        padding: 10px 4px 8px 4px;
        text-align: center;
        font-size: 14px;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .item-icon {
        font-size: 28px;
        margin-bottom: 4px;
    }
    @media screen and (max-width: 480px) {
        .app-card { padding: 12px 2vw 16px 2vw; }
        .title {
            font-size: clamp(4px, 1.5vw, 7px);
            white-space: normal;
        }
        .item-grid { gap: 8px; }
        .item-card { font-size: 12px; min-height: 60px; }
        .item-icon { font-size: 22px; }
    }
    </style>
    """, unsafe_allow_html=True)

# ラベルID→ゴミ種別・アイコン・説明のマッピング例
GARBAGE_LABEL_MAP = {
    # 例: ImageNetのラベルIDを仮で割り当て
    409: {  # "banana"
        'type': '可燃ゴミ',
        'icon': '🔥',
        'desc': ['皮や食べ残しは可燃ゴミです', '水気を切って捨てましょう']
    },
    569: {  # "plastic bag"
        'type': '資源ゴミ',
        'icon': '♻️',
        'desc': ['プラマークがある袋は資源ゴミです', '洗って乾かして出しましょう']
    },
    829: {  # "bottle"
        'type': '資源ゴミ',
        'icon': '🧴',
        'desc': ['ペットボトルは資源ゴミです', 'ラベルとキャップは外して']
    },
    569: {  # "plastic bag"
        'type': '資源ゴミ',
        'icon': '♻️',
        'desc': ['プラマークがある袋は資源ゴミです', '洗って乾かして出しましょう']
    },
    920: {  # "tin can"
        'type': '不燃ゴミ',
        'icon': '🗑️',
        'desc': ['缶は不燃ゴミです', '中を洗ってから捨てましょう']
    },
    # ... 必要に応じて追加 ...
}

def get_garbage_info(predicted_class):
    info = GARBAGE_LABEL_MAP.get(predicted_class)
    if info:
        return info['type'], info['icon'], info['desc']
    # デフォルト
    return 'その他', '❓', ['自治体のルールを確認してください']

# --- ページ切り替え ---
page = st.sidebar.radio('ページを選択', ['アシスタント', 'ごみ履歴'])

if 'garbage_history' not in st.session_state:
    st.session_state['garbage_history'] = []

if page == 'アシスタント':
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

    # タイトルと説明をカードで囲む
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div style="width: 100%; text-align: center;"><h1 class="title">♻️ ゴミ分別アシスタント ♻️</h1></div>', unsafe_allow_html=True)
    st.markdown("""
        ### 👋 こんにちは！ゴミ分別を手伝うよ！
        ゴミの写真を撮って、どこに捨てればいいか教えてあげるね！
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    # 画像アップロードもカードで囲む
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("### 📸 ゴミの写真を撮ってね！")
    uploaded_file = st.file_uploader("ここに写真をドラッグ＆ドロップするか、クリックして選んでね！", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(img_bytes))
        st.image(image, caption='📸 アップロードされた写真', use_column_width=True)
        with st.spinner('🔍 ゴミの種類を調べているよ...'):
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            garbage_type, garbage_icon, garbage_desc = get_garbage_info(predicted_class)
            st.markdown(f'<div class="result-text">このゴミは {garbage_icon} {garbage_type} です！</div>', unsafe_allow_html=True)
            st.markdown('<div class="garbage-info">', unsafe_allow_html=True)
            st.markdown("""
            ##### 💡 捨て方のポイント
            """)
            for d in garbage_desc:
                st.markdown(f'- {d}')
            st.markdown('</div>', unsafe_allow_html=True)
            # 履歴に追加（画像も保存）
            st.session_state['garbage_history'].append({
                'type': garbage_type,
                'icon': garbage_icon,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'img': img_bytes
            })
    st.markdown('</div>', unsafe_allow_html=True)

    # 地域情報の入力もカードで囲む
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("### 🏠 あなたの住んでいる場所を教えてね！")
    col1, col2 = st.columns(2)
    prefectures = load_location_data()
    with col1:
        prefecture = st.selectbox("都道府県", prefectures)
    with col2:
        city = st.text_input("市区町村", placeholder="例：渋谷区")
    if prefecture and city:
        schedule = get_garbage_schedule(prefecture, city)
        st.markdown('<div class="garbage-info">', unsafe_allow_html=True)
        st.markdown(f"### 📅 {prefecture}{city}のゴミ出しカレンダー")
        for garbage_type, days in schedule.items():
            if garbage_type != "注意事項":
                st.markdown(f"#### {garbage_type}")
                st.markdown(f"- {'・'.join(days)}")
        st.markdown("""
        ##### ⚠️ 注意事項
        """)
        for note in schedule["注意事項"]:
            st.markdown(f"- {note}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # フッターもカードで囲む
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("""
        ---
        ### 🌟 ゴミ分別で地球をきれいにしよう！
        """)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == 'ごみ履歴':
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div style="width: 100%; text-align: center;"><h1 class="title">🗂️ ごみ履歴</h1></div>', unsafe_allow_html=True)
    history = st.session_state['garbage_history']
    if history:
        st.markdown('<div class="item-grid">', unsafe_allow_html=True)
        import base64
        for item in reversed(history[-30:]):
            img_html = ''
            if 'img' in item and item['img']:
                img_b64 = base64.b64encode(item['img']).decode('utf-8')
                img_html = f'<img src="data:image/png;base64,{img_b64}" style="width:48px;height:48px;border-radius:8px;object-fit:cover;margin-bottom:4px;" />'
            st.markdown(f'<div class="item-card">{img_html}<div class="item-icon">{item["icon"]}</div>{item["type"]}<br><span style="font-size:10px;color:#888;">{item["time"]}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('まだごみは登録されていません。')
    st.markdown('</div>', unsafe_allow_html=True) 
