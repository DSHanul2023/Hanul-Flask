from get_data import get_item_data, get_chat_data, recommend_movies_for_members,minichatmovie
import torch
import os
from kogpt2_transformers import get_kogpt2_tokenizer
from model.kogpt2 import DialogKoGPT2, DialogKoGPT2Wrapper
from emotion import load_and_predict, load_c_model
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/survey": {"origins": "http://localhost:3000"}})


root_path = '.'
checkpoint_path = f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"

PATH = 'C:/Users/82109/Desktop/졸업 프로젝트/Flask-hanul/model/kobert_state_ver2.pt'

app = Flask(__name__)

tokenizer = get_kogpt2_tokenizer()

@app.route('/process', methods=['POST'])
def process_data():
    dialog_model = DialogKoGPT2Wrapper(os.path.abspath(save_ckpt_path), tokenizer)
    dialog_model.load_model()
    request_data = request.json
    question = request_data.get('question', '')
    answer = dialog_model.inference(question)
    response_data = {
        "answer": answer
    }

    # return response_data
    return answer

@app.route('/get_data', methods=['GET'])
def get_data():
    item_data = get_item_data()
    chat_data = get_chat_data()

    data = {
        "item_data": item_data,
        "chat_data": chat_data
    }

    return jsonify(data)

@app.route('/chatdata', methods=['GET'])
def get_chatdata():
    chat_data = get_chat_data()

    return jsonify({"chat_data": chat_data})

@app.route('/itemdata', methods=['GET'])
def get_itemdata():  # 함수 이름 변경
    item_data = get_item_data()

    return jsonify({"item_data": item_data})

@app.route('/movie', methods=['GET'])
def recommend_movies():
    item_data = get_item_data()
    chat_data = get_chat_data()

    recommended_movies = recommend_movies_for_members(item_data, chat_data)

    return jsonify(recommended_movies)

@app.route('/emotion', methods=['POST'])
def infer_emotion():
    request_data = request.json
    text = request_data.get('text', '')

    # 감정 분류 모델과 토크나이저 로드
    c_model, c_tokenizer = load_c_model(PATH)

    # 감정 분류 예측
    predicted_emotions = load_and_predict(text, c_model, c_tokenizer)

    response_data = {
        "predicted_emotions": predicted_emotions
    }

    return jsonify(response_data)

# 사용자의 선택 항목을 받아와 영화 추천을 처리
@app.route('/survey', methods=['POST'])
def minichatsurvey():
    try:
        request_data = request.get_json()

        # 클라이언트에서 전송한 선택 항목을 받아옴
        selected_emotions = request_data.get('selectedItems', [])

        # 감정 키워드에 해당하는 아이템을 가져오는 함수 호출
        recommended_movies = minichatmovie(selected_emotions)
        print("Recommended Movies:", recommended_movies)  # 이 부분을 추가

        # 추천된 영화를 JSON 형태로 반환
        return jsonify({"recommended_movies": recommended_movies})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
