# ゴミ分別アシスタント

このアプリケーションは、画像認識を使用してゴミの分別を支援するWebアプリケーションです。

## 機能

- ゴミの画像をアップロードして分別方法を確認
- 分別履歴の保存と管理
- ゴミ分別クイズ
- モバイル対応のレスポンシブデザイン

## セットアップ

1. 必要なパッケージをインストール:
```bash
pip install -r requirements.txt
```

2. アプリケーションを起動:
```bash
python app.py
```

3. ブラウザで以下のURLにアクセス:
```
http://localhost:8080
```

## 技術スタック

- Flask: Webフレームワーク
- PyTorch: 機械学習フレームワーク
- Transformers: 画像認識モデル
- Bootstrap: フロントエンドフレームワーク

## ライセンス

MITライセンス

## 使用方法

### 1. 環境構築
1. Pythonをインストール
   - [Python公式サイト](https://www.python.org/downloads/)からPythonをダウンロードしてインストール

2. 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

### 2. アプリケーションの起動
```bash
streamlit run app.py
```

### 3. アプリケーションの使用
1. ブラウザで表示されたURLにアクセス
2. 「ゴミの写真をアップロードしてください」から写真を選択
3. 都道府県と市区町村を入力
4. 結果を確認

## GitHubでの公開方法

1. GitHubアカウントの作成
   - [GitHub](https://github.com/)にアクセスしてアカウントを作成

2. リポジトリの作成
   - GitHubで新しいリポジトリを作成
   - リポジトリ名を入力（例：garbage-classifier）

3. コードのアップロード
```bash
git init
git add .
git commit -m "初回コミット"
git branch -M main
git remote add origin [リポジトリのURL]
git push -u origin main
```

4. Streamlit Cloudでの公開
   - [Streamlit Cloud](https://streamlit.io/cloud)にアクセス
   - GitHubアカウントでログイン
   - 「New app」をクリック
   - GitHubリポジトリを選択
   - デプロイを実行

## 注意事項
- 写真はjpg、jpeg、png形式に対応
- 地域情報は正確に入力してください 