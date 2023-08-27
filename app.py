from flask import Flask, request, jsonify

from get_data import get_item_data, get_chat_data, recommend_movies_for_members
import torch
import os
from kogpt2_transformers import get_kogpt2_tokenizer
from model.kogpt2 import DialogKoGPT2Wrapper
from emotion import predict

root_path = '.'
checkpoint_path = f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"

app = Flask(__name__)

tokenizer = get_kogpt2_tokenizer()

# 전역 변수로 모델을 저장할 변수
global dialog_model
dialog_model = None

@app.before_request # 처음 실행할 때 모델 한 번만 load
def load_model():
    global dialog_model
    dialog_model = DialogKoGPT2Wrapper(os.path.abspath(checkpoint_path), tokenizer)
    dialog_model.load_model()

@app.route('/process', methods=['POST'])
def process_data():
    global dialog_model

    request_data = request.json
    question = request_data.get('question', '')
    answer = dialog_model.inference(question)

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
def emotion():
    request_data = request.json
    sentence = request_data.get('sentence', '')
    result = predict(sentence)
    print(">> 입력하신 내용에서 " + result + " 느껴집니다.")
    return result



if __name__ == '__main__':
    app.run()

