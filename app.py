import streamlit as st
import requests
from PIL import Image
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import json
import pandas as pd
from datetime import datetime
import hashlib
import base64
import os
import random

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

# クイズデータ
QUIZ_DATA = [
    {
        'question': 'ペットボトルのキャップは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'ペットボトルのキャップはプラスチック製で、リサイクル可能な資源ゴミです。'
    },
    {
        'question': '使用済みのティッシュは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': 'ティッシュは紙製で、燃やせるゴミとして処理されます。'
    },
    {
        'question': 'アルミ缶は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'アルミ缶はリサイクル可能な資源ゴミです。'
    },
    {
        'question': 'スプレー缶は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'スプレー缶は中身を完全に使い切ってから不燃ゴミとして出します。'
    },
    {
        'question': '牛乳パックは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': '牛乳パックは洗って乾かしてから資源ゴミとして出します。'
    },
    {
        'question': '使用済みの油は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': 'その他',
        'explanation': '使用済みの油は固めてから可燃ゴミとして出します。'
    },
    {
        'question': 'CDやDVDは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'CDやDVDは不燃ゴミとして出します。'
    },
    {
        'question': '新聞紙は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': '新聞紙は資源ゴミとして出します。'
    },
    {
        'question': '生ゴミは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '生ゴミは水気を切ってから可燃ゴミとして出します。'
    },
    {
        'question': '蛍光灯は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': '蛍光灯は不燃ゴミとして出します。'
    },
    {
        'question': '乾電池は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': '乾電池は不燃ゴミとして出します。'
    },
    {
        'question': '段ボールは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': '段ボールは資源ゴミとして出します。'
    },
    {
        'question': '衣類は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '衣類は可燃ゴミとして出します。'
    },
    {
        'question': 'ガラス瓶は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'ガラス瓶は資源ゴミとして出します。'
    },
    {
        'question': '傘は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': '傘は不燃ゴミとして出します。'
    },
    {
        'question': '紙パックは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': '紙パックは洗って乾かしてから資源ゴミとして出します。'
    },
    {
        'question': 'プラスチック容器は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'プラスチック容器は洗って乾かしてから資源ゴミとして出します。'
    },
    {
        'question': 'おもちゃは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'おもちゃは不燃ゴミとして出します。'
    },
    {
        'question': 'カセットテープは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'カセットテープは不燃ゴミとして出します。'
    },
    {
        'question': 'マグカップは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'マグカップは不燃ゴミとして出します。'
    },
    {
        'question': 'アルミホイルは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'アルミホイルは不燃ゴミとして出します。'
    },
    {
        'question': 'ティッシュの箱は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'ティッシュの箱は資源ゴミとして出します。'
    },
    {
        'question': '使い捨てカイロは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': '使い捨てカイロは不燃ゴミとして出します。'
    },
    {
        'question': '紙コップは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙コップは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのハンガーは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのハンガーは不燃ゴミとして出します。'
    },
    {
        'question': '紙袋は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙袋は可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのストローは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのストローは不燃ゴミとして出します。'
    },
    {
        'question': '紙の封筒は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙の封筒は可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのフォークは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのフォークは不燃ゴミとして出します。'
    },
    {
        'question': '紙のレシートは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙のレシートは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのスプーンは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのスプーンは不燃ゴミとして出します。'
    },
    {
        'question': '紙の名刺は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙の名刺は可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックの箸は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックの箸は不燃ゴミとして出します。'
    },
    {
        'question': '紙の包装紙は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙の包装紙は可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックの容器のフタは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'プラスチックの容器のフタは資源ゴミとして出します。'
    },
    {
        'question': '紙の箱は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': '紙の箱は資源ゴミとして出します。'
    },
    {
        'question': 'プラスチックのボトルは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'プラスチックのボトルは資源ゴミとして出します。'
    },
    {
        'question': '紙のカレンダーは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙のカレンダーは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのバッグは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'プラスチックのバッグは資源ゴミとして出します。'
    },
    {
        'question': '紙のノートは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙のノートは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのケースは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのケースは不燃ゴミとして出します。'
    },
    {
        'question': '紙のチラシは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙のチラシは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのトレイは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'プラスチックのトレイは資源ゴミとして出します。'
    },
    {
        'question': '紙のポスターは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙のポスターは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのカップは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのカップは不燃ゴミとして出します。'
    },
    {
        'question': '紙のメモは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙のメモは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのボールは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのボールは不燃ゴミとして出します。'
    },
    {
        'question': '紙の写真は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙の写真は可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックの玩具は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックの玩具は不燃ゴミとして出します。'
    },
    {
        'question': '紙のカードは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙のカードは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックの容器は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'プラスチックの容器は資源ゴミとして出します。'
    },
    {
        'question': '紙の本は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙の本は可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックの袋は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '資源ゴミ',
        'explanation': 'プラスチックの袋は資源ゴミとして出します。'
    },
    {
        'question': '紙の雑誌は何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙の雑誌は可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのボックスは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのボックスは不燃ゴミとして出します。'
    },
    {
        'question': '紙のカタログは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙のカタログは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのバケツは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのバケツは不燃ゴミとして出します。'
    },
    {
        'question': '紙のカレンダーは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '可燃ゴミ',
        'explanation': '紙のカレンダーは可燃ゴミとして出します。'
    },
    {
        'question': 'プラスチックのバスケットは何ゴミですか？',
        'options': ['可燃ゴミ', '不燃ゴミ', '資源ゴミ', 'その他'],
        'correct': '不燃ゴミ',
        'explanation': 'プラスチックのバスケットは不燃ゴミとして出します。'
    }
]

# スタンプデータ
STAMPS = {
    '可燃ゴミ': '🔥',
    '不燃ゴミ': '🗑️',
    '資源ゴミ': '♻️',
    'その他': '❓'
}

# --- ページ切り替え ---
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'アシスタント'

# サイドバーのページ選択
page = st.sidebar.radio('ページを選択', ['アシスタント', 'ごみ履歴', 'ごみクイズ'], key='page_selector')

if 'garbage_history' not in st.session_state:
    st.session_state['garbage_history'] = []

def get_image_hash(img_bytes):
    return hashlib.md5(img_bytes).hexdigest()

def is_duplicate_image(img_bytes, history):
    img_hash = get_image_hash(img_bytes)
    for item in history:
        if 'img_hash' in item and item['img_hash'] == img_hash:
            return True
    return False

# 効果音のBase64エンコードされたデータ
CORRECT_SOUND = """
UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU
"""

INCORRECT_SOUND = """
UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU
"""

# 効果音の再生関数
def play_sound(sound_type):
    try:
        if sound_type == 'correct':
            audio_b64 = CORRECT_SOUND
        elif sound_type == 'incorrect':
            audio_b64 = INCORRECT_SOUND
        else:
            with open('break.mp3', 'rb') as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            
        audio_html = f'''
            <audio autoplay hidden>
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
        '''
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"効果音の再生に失敗しました: {str(e)}")

# クイズの問題をシャッフルする関数
def shuffle_quiz():
    random.shuffle(QUIZ_DATA)

if page == 'アシスタント':
    # モデルの読み込み
    @st.cache_resource
    def load_model():
        try:
            # より軽量なモデルを使用
            processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
            model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
            return processor, model
        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {str(e)}")
            # フォールバックとしてダミーモデルを返す
            return None, None

    # 画像分類関数
    def classify_image(image, processor, model):
        try:
            if processor is None or model is None:
                return 409  # デフォルトのラベルIDを返す
            
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()
            return predicted_class
        except Exception as e:
            st.error(f"画像分類中にエラーが発生しました: {str(e)}")
            return 409  # エラー時はデフォルトのラベルIDを返す

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
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='📸 アップロードされた写真', use_column_width=True)
            
            # モデルの読み込み
            processor, model = load_model()
            
            # 画像分類
            predicted_class = classify_image(image, processor, model)
            garbage_type, garbage_icon, garbage_desc = get_garbage_info(predicted_class)
            
            # 結果の表示
            st.markdown(f'<div class="result-text">このゴミは {garbage_icon} {garbage_type} です！</div>', unsafe_allow_html=True)
            play_sound('correct')  # 正解時の効果音

            # 分別方法の詳細リンクを表示
            if 'prefecture' in locals() and 'city' in locals() and prefecture and city:
                search_url = f'https://www.google.com/search?q={prefecture}+{city}+ごみ分別'
                st.markdown(f'<a href="{search_url}" target="_blank">🔎 {prefecture}{city}の分別方法を検索する</a>', unsafe_allow_html=True)

            st.markdown('<div class="garbage-info">', unsafe_allow_html=True)
            st.markdown("""
            ##### 💡 捨て方のポイント
            """)
            for d in garbage_desc:
                st.markdown(f'- {d}')
            st.markdown('</div>', unsafe_allow_html=True)
            # 履歴に追加（画像も保存）
            if not is_duplicate_image(uploaded_file.getvalue(), st.session_state['garbage_history']):
                st.session_state['garbage_history'].append({
                    'type': garbage_type,
                    'icon': garbage_icon,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'img': uploaded_file.getvalue(),
                    'img_hash': get_image_hash(uploaded_file.getvalue())
                })
        except Exception as e:
            st.error(f"画像処理中にエラーが発生しました: {str(e)}")
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
    st.title('ごみ履歴')
    
    # 履歴の表示と操作
    if st.session_state['garbage_history']:
        # 全削除ボタン
        if st.button('履歴を全削除'):
            st.session_state['garbage_history'] = []
            st.success('履歴を全削除しました')
            st.rerun()
        
        # 履歴の表示
        for i, item in enumerate(st.session_state['garbage_history']):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"日時: {item['time']}")
                st.write(f"種類: {item['type']}")
                if 'img' in item:
                    img_b64 = base64.b64encode(item['img']).decode('utf-8')
                    st.image(f'data:image/png;base64,{img_b64}', width=200)
            
            with col2:
                if st.button('削除', key=f'delete_{i}'):
                    delete_history_item(i)
            
            with col3:
                if st.button('スタンプ', key=f'stamp_{i}'):
                    st.balloons()
                    st.success(f'スタンプを獲得しました！ {STAMPS.get(item["type"], "❓")}')
            
            st.divider()
        
        # CSVダウンロードボタン
        if st.button('履歴をCSVでダウンロード'):
            df = pd.DataFrame(st.session_state['garbage_history'])
            csv = df.to_csv(index=False)
            st.download_button(
                label="CSVファイルをダウンロード",
                data=csv,
                file_name="garbage_history.csv",
                mime="text/csv"
            )
    else:
        st.info('履歴がありません')

elif page == 'ごみクイズ':
    st.title('ごみ分別クイズ')
    
    if 'quiz_index' not in st.session_state:
        st.session_state['quiz_index'] = 0
        st.session_state['score'] = 0
        shuffle_quiz()  # クイズをシャッフル
    
    current_quiz = QUIZ_DATA[st.session_state['quiz_index']]
    
    st.write(f"問題 {st.session_state['quiz_index'] + 1}/{len(QUIZ_DATA)}")
    st.write(current_quiz['question'])
    
    selected_option = st.radio('選択してください', current_quiz['options'])
    
    if st.button('回答'):
        if selected_option == current_quiz['correct']:
            st.success('正解です！')
            st.session_state['score'] += 1
            st.balloons()
            st.audio('correct.mp3', format='audio/mp3', start_time=0)
        else:
            st.error('不正解です')
            st.audio('incorrect.mp3', format='audio/mp3', start_time=0)
        st.write(f"解説: {current_quiz['explanation']}")
        
        if st.session_state['quiz_index'] < len(QUIZ_DATA) - 1:
            if st.button('次の問題へ'):
                st.session_state['quiz_index'] += 1
                st.rerun()
        else:
            st.write(f"クイズ終了！ スコア: {st.session_state['score']}/{len(QUIZ_DATA)}")
            if st.button('クイズをリセット'):
                st.session_state['quiz_index'] = 0
                st.session_state['score'] = 0
                shuffle_quiz()  # クイズを再度シャッフル
                st.rerun()

# 履歴の削除機能を修正
def delete_history_item(index):
    if 0 <= index < len(st.session_state['garbage_history']):
        st.session_state['garbage_history'].pop(index)
        st.success('履歴を削除しました')
        st.rerun() 