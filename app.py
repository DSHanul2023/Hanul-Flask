from flask import Flask, request, jsonify
from kobert_tokenizer import KoBERTTokenizer

import emotion
from get_data import get_item_data, get_chat_data, recommend_movies_for_members
import torch
import os
from kogpt2_transformers import get_kogpt2_tokenizer
from model.kogpt2 import DialogKoGPT2, DialogKoGPT2Wrapper
from emotion import BERTClassifier,predict
root_path = '.'
checkpoint_path = f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"
save_ckpt_path2 = f"{checkpoint_path}/quantized_kogpt2-wellnesee-auto-regressive.pth"
app = Flask(__name__)
ctx = "cpu"

tokenizer = get_kogpt2_tokenizer()


@app.route('/process2',methods=['POST'])
def process2_data():
    loaded_quantized_model = DialogKoGPT2Wrapper(os.path.abspath(save_ckpt_path2), tokenizer)
    loaded_quantized_model.load_model()
    request_data = request.json
    question = request_data.get('question', '')
    answer = dialog_model.inference(question)
    return answer

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
def process_emotion():
    request_data = request.json
    question = request_data.get('question', '')
    result = predict(question)
    response_data = {
        "predicted_emotion": result
    }
    return jsonify(response_data)
if __name__ == '__main__':
    dialog_model = DialogKoGPT2Wrapper(os.path.abspath(save_ckpt_path), tokenizer)
    dialog_model.load_model()
    app.run()