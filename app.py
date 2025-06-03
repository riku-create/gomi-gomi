import streamlit as st
import requests
from PIL import Image
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# ページ設定
st.set_page_config(page_title="ゴミ分別アシスタント", layout="wide")
st.title("ゴミ分別アシスタント")

# モデルの読み込み
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    return processor, model

processor, model = load_model()

# 画像アップロード
uploaded_file = st.file_uploader("ゴミの写真をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_column_width=True)
    
    # 画像の分類
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # 分類結果の表示（ここでは仮の結果を表示）
    st.write("分類結果：可燃ゴミ")

# 地域情報の入力
st.header("地域情報の入力")
prefecture = st.text_input("都道府県を入力してください")
city = st.text_input("市区町村を入力してください")

if prefecture and city:
    # ここで実際のAPIを呼び出してゴミ出し日程を取得
    # 仮のデータを表示
    st.write(f"{prefecture}{city}のゴミ出し日程")
    st.write("可燃ゴミ：毎週月曜日・木曜日")
    st.write("不燃ゴミ：毎週火曜日")
    st.write("資源ゴミ：毎週金曜日") 