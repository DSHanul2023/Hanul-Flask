from emotion import predict
from get_data import get_chat, preprocess_text, get_view
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import mysql.connector

# MySQL 데이터베이스 연결 설정
db_config = {
    "host": "127.0.0.1",  # 호스트 주소
    "user": "root",       # 사용자 이름
    "password": "hanul",  # 비밀번호
    "database": "hanuldb",  # 데이터베이스 이름
    "port": 3306          # MySQL 포트 번호
}

# 추천 영화 중 중복 제거
def remove_duplicate_movies(movies):
    unique_movies = []
    movie_ids_seen = set()

    for movie in movies:
        if movie['movie_id'] not in movie_ids_seen:
            unique_movies.append(movie)
            movie_ids_seen.add(movie['movie_id'])

    return unique_movies


# 감정별 뷰 생성
def create_view():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    query = "CREATE VIEW anger AS SELECT * FROM item WHERE genre_name LIKE '%액션%' AND genre_name LIKE '%범죄%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW worry AS SELECT * FROM item WHERE genre_name LIKE '%드라마%' AND genre_name LIKE '%로맨스%' AND genre_name LIKE '%가족%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW anxiety AS SELECT * FROM item WHERE genre_name LIKE '%로맨스%' AND genre_name LIKE '%드라마%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW depression AS SELECT * FROM item WHERE genre_name LIKE '%드라마%' AND genre_name LIKE '%음악%' AND genre_name LIKE '%코미디%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW fear AS SELECT * FROM item WHERE genre_name LIKE '%애니메이션%' AND genre_name LIKE '%가족%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW sad AS SELECT * FROM item WHERE genre_name LIKE '%드라마%' AND genre_name LIKE '%애니메이션%' AND genre_name LIKE '%코미디%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW joy AS SELECT * FROM item WHERE genre_name LIKE '%판타지%' AND genre_name LIKE '%모험%' AND genre_name LIKE '%액션%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW neutral AS SELECT * FROM item WHERE genre_name LIKE '%sf%' AND genre_name LIKE '%모험%'"
    cursor.execute(query.encode('utf8'))


# 사용자 발화와 영화 내용 간 가중치 계산
def calc_weight(item_data, chat_data):
    # 영화 정보 데이터
    movie_info = [{'item_id': item[0], 'genre': item[1], 'description': item[2], 'title': item[3], 'movie_id': item[4],
                   'image_url': item[5], 'member_id': item[6], 'tokens': item[7]} for item in item_data]

    # 사용자 발화 하나의 문자열로 합치기
    delimiter = " "
    user_says = delimiter.join(chat_data)

    # 유사도 계산
    tokens = [preprocess_text(movie['tokens']) for movie in movie_info]
    tokens.append(user_says)

    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(tokens)

    # 사용자 발화 벡터를 만들어서 코사인 유사도 계산
    user_says_list = [user_says]
    print(user_says_list)
    cosine_similarities = cosine_similarity(vectorizer.transform(user_says_list), vectorized_data).flatten()

    tokens.pop()

    for idx, sim in enumerate(cosine_similarities):
        if idx < len(movie_info):
            movie_info[idx]['cosine_similarity'] = sim
        else:
            print(f"Index {idx} is out of range for movie_info list.")

    return movie_info

def recommendation(user_id):
    chat_data = get_chat(user_id)
    predicted_emotions = []
    user_says = []

    # 사용자 발화 감정 분석
    for text in chat_data:
        predicted_emotions.append(predict(text))
        user_says.append(preprocess_text(text))

    # 빈출 감정 선택
    count_emotion = Counter(predicted_emotions)
    emotions = (count_emotion.most_common(n=1))
    emotion = emotions[0][0]

    e_view = ""

    if emotion == 0:
        e_view = 'anger'
    elif emotion == 1:
        e_view = 'sad'
    elif emotion == 2:
        e_view = 'joy'
    elif emotion == 3:
        e_view = 'worry'
    elif emotion == 4:
        e_view = 'anxiety'
    elif emotion == 5:
        e_view = 'neutral'
    elif emotion == 6:
        e_view = 'depression'
    elif emotion == 7:
        e_view = 'fear'


    # 분류된 감정에 맞는 view에서 아이템 불러오기
    item_data = get_view(e_view)
    recommend_movies = calc_weight(item_data, user_says)

    cosine_similarity_list=[]
    for movie in recommend_movies:
        cosine_similarity_list.append([movie['cosine_similarity']])
    print(sorted(cosine_similarity_list))

    return recommend_movies

def recommend_movies_for_members(pre_item_data, item_data, chat_data):
    # 영화 정보 데이터
    movie_info = [{'item_id': item[0], 'genre': item[1], 'description': item[2], 'title': item[3], 'movie_id': item[4],
                   'image_url': item[5], 'member_id': item[6], 'tokens': item[7]} for item in item_data]

    # 각 멤버의 채팅 메시지를 저장할 딕셔너리
    member_chat_messages = {}

    # 채팅 데이터를 멤버별로 그룹화하여 처리
    for chat in chat_data:
        member_id, message = chat[3], chat[1]
        if member_id not in member_chat_messages:
            member_chat_messages[member_id] = []
        member_chat_messages[member_id].append(message)

    # 멤버별 추천 영화 정보를 저장하는 딕셔너리
    recommended_movies = {}

    # 멤버별 채팅 데이터를 가져와서 처리
    chat_messages = member_chat_messages.get(member_id, [])

    start_time = time.time()  # 시작 시간 기록
    # 데이터 전처리
    preprocessed_chat_messages = [preprocess_text(text) for text in chat_messages]

    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 수행 시간 계산
    print(f"데이터 전처리 시간: {elapsed_time:.4f} seconds")  # 수행 시간 출력

    start_time = time.time()  # 시작 시간 기록
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_chat_messages + pre_item_data)
    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 수행 시간 계산
    print(f"TF-IDF 벡터화 시간: {elapsed_time:.4f} seconds")  # 수행 시간 출력

    start_time = time.time()  # 시작 시간 기록
    # 채팅 메시지와 영화 정보 간의 코사인 유사도 계산
    similarity_matrix = cosine_similarity(tfidf_matrix)
    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 수행 시간 계산
    print(f"코사인 유사도 계산 시간: {elapsed_time:.4f} seconds")  # 수행 시간 출력

    # 유사도가 높은 영화 추천
    chat_similarity_scores = similarity_matrix[:-len(movie_info), -len(movie_info):]
    top_similar_indices = chat_similarity_scores.argmax(axis=1)
    print(chat_similarity_scores)
    print(top_similar_indices)

    recommended_movies = [movie_info[idx] for idx in top_similar_indices]

    # 중복 영화 제거
    recommended_movies = remove_duplicate_movies(recommended_movies)

    return recommended_movies

if __name__ == "__main__":
    # create_view()
    recommendation('402899838a23ffb6018a2400f47504f6')

