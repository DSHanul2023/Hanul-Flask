from flask import Flask, request, jsonify

from get_data import get_item_data, get_chat_data, recommend_movies_for_members, preprocess_movie_info
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

# dialog_model 미리 로드 - load_model 속도 개선
global dialog_model
dialog_model = DialogKoGPT2Wrapper(os.path.abspath(save_ckpt_path), tokenizer)
dialog_model.load_model()
print("load_model 실행됨")

# movie detail 미리 전처리 - 전처리 속도 개선
global pre_item_data
pre_item_data = None
global item_data
item_data = get_item_data()
movie_info = [{'item_id': item[0], 'genre': item[1], 'description': item[2], 'title': item[3], 'movie_id': item[4],
               'image_url': item[5], 'member_id': item[6]} for item in item_data]
pre_item_data = preprocess_movie_info(movie_info)
print("preprocess_item 실행됨")

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

import time
@app.route('/movie', methods=['POST'])
def recommend_movies():
    global pre_item_data
    global item_data
    request_data = request.json
    member_id = request_data.get('member_id', '')

    chat_data = get_chat_data(member_id)

    start_time = time.time()  # 시작 시간 기록
    recommended_movies = recommend_movies_for_members(pre_item_data, item_data, chat_data)
    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 수행 시간 계산
    print(f"recommend_movies 총 시간: {elapsed_time:.4f} seconds")  # 수행 시간 출력

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

