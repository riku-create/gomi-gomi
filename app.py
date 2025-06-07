from flask import Flask, render_template, request, jsonify, session, send_file, send_from_directory
from PIL import Image
import io
import base64
import random
import hashlib
from datetime import datetime
import pandas as pd
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

app = Flask(__name__, template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = os.urandom(24)  # セッション用の秘密鍵

# ラベルID→ゴミ種別・アイコン・説明のマッピング例
GARBAGE_LABEL_MAP = {
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
    920: {  # "tin can"
        'type': '不燃ゴミ',
        'icon': '🗑️',
        'desc': ['缶は不燃ゴミです', '中を洗ってから捨てましょう']
    }
}

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
    }
]

def get_garbage_info(predicted_class):
    info = GARBAGE_LABEL_MAP.get(predicted_class)
    if info:
        return info['type'], info['icon'], info['desc']
    return 'その他', '❓', ['自治体のルールを確認してください']

def get_image_hash(img_bytes):
    return hashlib.md5(img_bytes).hexdigest()

def is_duplicate_image(img_bytes, history):
    img_hash = get_image_hash(img_bytes)
    for item in history:
        if 'img_hash' in item and item['img_hash'] == img_hash:
            return True
    return False

def load_model():
    try:
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
        return processor, model
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {str(e)}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assistant')
def assistant():
    return render_template('assistant.html')

@app.route('/history')
def history():
    if 'garbage_history' not in session:
        session['garbage_history'] = []
    return render_template('history.html', history=session['garbage_history'])

@app.route('/quiz')
def quiz():
    if 'quiz_state' not in session:
        session['quiz_state'] = {
            'score': 0,
            'current_question': 0,
            'started': False
        }
    return render_template('quiz.html', 
                         quiz_data=QUIZ_DATA,
                         quiz_state=session['quiz_state'])

@app.route('/api/quiz/start', methods=['POST'])
def start_quiz():
    session['quiz_state'] = {
        'score': 0,
        'current_question': 0,
        'started': True
    }
    session.modified = True
    return jsonify({'success': True})

@app.route('/api/quiz/restart', methods=['POST'])
def restart_quiz():
    session['quiz_state'] = {
        'score': 0,
        'current_question': 0,
        'started': True
    }
    session.modified = True
    return jsonify({'success': True})

@app.route('/api/quiz/check', methods=['POST'])
def check_answer():
    data = request.get_json()
    question_index = data.get('question_index')
    answer = data.get('answer')
    
    if question_index is None or answer is None:
        return jsonify({'error': 'パラメータが不足しています'}), 400
    
    question = QUIZ_DATA[question_index]
    is_correct = answer == question['correct']
    
    if is_correct:
        if 'quiz_state' not in session:
            session['quiz_state'] = {'score': 0, 'current_question': 0, 'started': True}
        session['quiz_state']['score'] += 1
        session.modified = True
    
    return jsonify({
        'is_correct': is_correct,
        'explanation': question['explanation'],
        'score': session.get('quiz_state', {}).get('score', 0)
    })

@app.route('/api/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': '画像がアップロードされていません'}), 400
    
    try:
        image_file = request.files['image']
        image = Image.open(image_file)
        
        processor, model = load_model()
        if processor and model:
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()
            
            garbage_type, garbage_icon, garbage_desc = get_garbage_info(predicted_class)
            
            # 履歴に追加
            if 'garbage_history' not in session:
                session['garbage_history'] = []
            
            image_bytes = image_file.read()
            if not is_duplicate_image(image_bytes, session['garbage_history']):
                session['garbage_history'].append({
                    'type': garbage_type,
                    'icon': garbage_icon,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'img': base64.b64encode(image_bytes).decode('utf-8'),
                    'img_hash': get_image_hash(image_bytes)
                })
                session.modified = True
            
            return jsonify({
                'type': garbage_type,
                'icon': garbage_icon,
                'description': garbage_desc,
                'play_sound': True
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/delete', methods=['POST'])
def delete_history():
    data = request.get_json()
    index = data.get('index')
    
    if index is not None and 'garbage_history' in session:
        if 0 <= index < len(session['garbage_history']):
            session['garbage_history'].pop(index)
            session.modified = True
            return jsonify({'success': True})
    
    return jsonify({'error': '削除に失敗しました'}), 400

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    session['garbage_history'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/api/history/download')
def download_history():
    if 'garbage_history' not in session:
        return jsonify({'error': '履歴がありません'}), 404
    
    df = pd.DataFrame(session['garbage_history'])
    csv = df.to_csv(index=False)
    
    return send_file(
        io.BytesIO(csv.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='garbage_history.csv'
    )

@app.route('/static/sounds/<path:filename>')
def serve_sound(filename):
    return send_from_directory('static/sounds', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)