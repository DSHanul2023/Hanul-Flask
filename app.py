from flask import Flask, request, jsonify
from get_data import get_item, get_chat
from add_tokens import mecab_preprocess
from minichat import minichatmovie
import os
from recommend import create_view, recommendation
from kogpt2_transformers import get_kogpt2_tokenizer
from model.kogpt2 import DialogKoGPT2Wrapper
from emotion import predict
from flask import Flask, request, jsonify
from flask_cors import CORS
from recommend import create_view
from add_tokens import mecab_preprocess
import requests


root_path = '.'
checkpoint_path = f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"

# save_ckpt_path2 = f"{checkpoint_path}/quantized_kogpt2-wellnesee-auto-regressive.pth"

app = Flask(__name__)
CORS(app, resources={r"/survey": {"origins": "http://localhost:3000"}})
# CORS(app, resources={r"/recommend": {"origins": "http://localhost:3000"}})
ctx = "cpu"

tokenizer = get_kogpt2_tokenizer()

# dialog_model 미리 로드 - load_model 속도 개선
global dialog_model
dialog_model = DialogKoGPT2Wrapper(os.path.abspath(save_ckpt_path), tokenizer)
dialog_model.load_model()
print("load_model 실행됨")

'''
# global loaded_quantized_model
# loaded_quantized_model = DialogKoGPT2Wrapper(os.path.abspath(save_ckpt_path2), tokenizer)
# loaded_quantized_model.load_model()
# print("loaded_quantized_model 실행됨")

# movie detail 미리 전처리 - 전처리 속도 개선
global pre_item_data
pre_item_data = None
global item_data
item_data = get_item()
movie_info = [{'item_id': item[0], 'genre': item[1], 'description': item[2], 'title': item[3], 'movie_id': item[4],
               'image_url': item[5], 'member_id': item[6]} for item in item_data]
pre_item_data = preprocess_movie_info(movie_info)
print("preprocess_item 실행됨")
'''

# 영화 데이터 토큰화
# mecab_preprocess()

# 감정 뷰 생성 (2. 처음 한 번 실행)
# create_view()

# 모델 

#@app.route('/process2',methods=['POST'])
#def process2_data():
    #global loaded_quantized_model

    #request_data = request.json
    #question = request_data.get('question', '')
    #answer = loaded_quantized_model.inference(question)
    #return answer

@app.route('/process', methods=['POST'])
def process_data():
    global dialog_model

    request_data = request.json
    question = request_data.get('question', '')
    answer = dialog_model.inference(question)

    return answer

@app.route('/get_data', methods=['GET'])
def get_data():
    item_data = get_item()
    chat_data = get_chat()

    data = {
        "item_data": item_data,
        "chat_data": chat_data
    }

    return jsonify(data)

# 북마크한 아이템 가져오기
@app.route('/getsaveditems')
def get_spring_data():
    memberId = request.args.get('memberId')

    spring_url = f"http://localhost:8080/members/{memberId}/bookmarked-items"  # Replace with your memberId
    response = requests.get(spring_url)

    if response.status_code == 200:
        spring_data = response.json()  # Assuming the response is in JSON format
        # Process the spring_data here
        return jsonify(spring_data)
    else:
        return "Error fetching data from Spring Boot", response.status_code

@app.route('/recommend', methods=['POST'])
def recommend_movie():
    request_data = request.json
    user_id = request_data.get('user_id', '')
    saved = request_data.get('saved', '')

    recommended = recommendation(user_id, saved)

    return recommended

# 테스트 : /recommend2?memberId={memberId}
@app.route('/recommend2', methods=['GET'])
def recommend_movie2():
    memberId = request.args.get('memberId')
    spring_url = f"http://localhost:8080/members/{memberId}/bookmarked-items"
    response = requests.get(spring_url)
    saved_data = response.json()
    saved = [item["id"] for item in saved_data]

    recommended = recommendation(memberId, saved)

    return recommended


@app.route('/chatdata', methods=['GET'])
def get_chatdata():
    chat_data = get_chat()

    return jsonify({"chat_data": chat_data})

@app.route('/itemdata', methods=['GET'])
def get_itemdata():  # 함수 이름 변경
    item_data = get_item()

    return jsonify({"item_data": item_data})


import time
'''
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
'''

@app.route('/emotion', methods=['POST'])

def emotion():
    request_data = request.json
    sentence = request_data.get('sentence', '')
    result = predict(sentence)
    print(">> 입력하신 내용에서 " + result + " 느껴집니다.")
    return result

def process_emotion():
    request_data = request.json
    question = request_data.get('question', '')
    result = predict(question)
    response_data = {
        "predicted_emotion": result
    }
    return jsonify(response_data)



# 사용자의 선택 항목을 받아와 영화 추천을 처리
@app.route('/survey', methods=['POST'])
def minichatsurvey():
    try:
        request_data = request.get_json()

        # 클라이언트에서 전송한 선택 항목을 받아옴
        selected_emotions = request_data.get('selectedItems', [])
        print(selected_emotions)
        # selected_genres = request_data.get('genres', [])

        # 감정 키워드에 해당하는 영화 추천
        recommended_movies_emotion = minichatmovie(selected_emotions)
        print(recommended_movies_emotion)

        # 장르 키워드에 해당하는 영화 추천
        # recommended_movies_genre = minichatmovie(selected_genres)
        # print(recommended_movies_genre)

        # 감정과 장르에 따른 추천 영화를 병합하여 최종 추천 리스트 생성
        #final_recommended_movies = recommended_movies_emotion + recommended_movies_genre

        # 중복 영화 제거
        #final_recommended_movies = remove_duplicate_movies(final_recommended_movies)
        recommended_movies_emotion

        # 추천된 영화를 JSON 형태로 반환
        return jsonify({"recommended_movies": recommended_movies_emotion})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()
