from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
import mysql.connector

# MySQL 데이터베이스 연결 설정
db_config = {
    "host": "127.0.0.1",  # 호스트 주소
    "user": "root",       # 사용자 이름
    "password": "hanul",  # 비밀번호
    "database": "hanuldb",  # 데이터베이스 이름
    "port": 3306          # MySQL 포트 번호
}

# KoNLPy의 Okt 객체 생성
okt = Okt()

# 텍스트 전처리 및 토큰화 함수
def preprocess_text(text):
    words = okt.morphs(text, stem=True)
    return ' '.join(words)

# "item" 테이블 데이터 가져오기
def get_item_data():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    query = "SELECT * FROM item"
    cursor.execute(query)

    item_data = cursor.fetchall()

    cursor.close()
    connection.close()

    return item_data

# "chat" 테이블 데이터 가져오기
def get_chat_data(member_id):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    query = "SELECT * FROM chat WHERE member_id = %s"
    cursor.execute(query, (member_id,))  # 매개변수를 통해 SQL 쿼리 파라미터 전달

    chat_data = cursor.fetchall()

    cursor.close()
    connection.close()

    return chat_data

# 미니 챗 영화 추천
def minichatmovie(selected_emotions):
    # 각 감정에 해당하는 영화 장르 정의
    emotion_to_genre = {
        '분노': ['액션', '범죄'],
        '걱정': ['드라마', '로맨스', '가족'],
        '불안감': ['로맨스', '드라마'],
        '우울감': ['드라마', '음악', '코미디'],
        '공포': ['애니메이션', '가족'],
        '슬픔': ['드라마', '애니메이션', '코미디'],
        '기쁨': ['판타지', '모험', '액션'],
        '설렘': ['SF', '모험']
    }
    
    selected_genres = []
    for emotion in selected_emotions:
        selected_genres.extend(emotion_to_genre.get(emotion, []))
    
    # 감정 키워드에 해당하는 아이템을 데이터베이스에서 가져오는 로직
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    genre_placeholders = ', '.join(['%s'] * len(selected_genres))
    query = f"SELECT * FROM item WHERE genre_name IN ({genre_placeholders})"
    cursor.execute(query.encode('utf8'), tuple(selected_genres))
    recommended_movies = cursor.fetchall()

    cursor.close()
    conn.close()

    return recommended_movies

# 추천 영화 중 중복 제거
def remove_duplicate_movies(movies):
    unique_movies = []
    movie_ids_seen = set()

    for movie in movies:
        if movie['movie_id'] not in movie_ids_seen:
            unique_movies.append(movie)
            movie_ids_seen.add(movie['movie_id'])

    return unique_movies

def preprocess_movie_info(movie_info):
    preprocessed_movie_info = [preprocess_text(f"{info['title']} {info['description']} {info['genre']}") for info in movie_info]
    return preprocessed_movie_info

import time

def recommend_movies_for_members(pre_item_data, item_data, chat_data):
    # 영화 정보 데이터
    movie_info = [{'item_id': item[0], 'genre': item[1], 'description': item[2], 'title': item[3], 'movie_id': item[4],
                   'image_url': item[5], 'member_id': item[6]} for item in item_data]

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
    for member_id in member_ids:
        chat_messages = member_chat_messages.get(member_id, [])

        # 데이터 전처리
        preprocessed_chat_messages = [preprocess_text(text) for text in chat_messages]
        preprocessed_movie_info = [preprocess_text(f"{info['title']} {info['description']} {info['genre']}") for info in movie_info if info['member_id'] != member_id]

        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_chat_messages + preprocessed_movie_info)

        # 채팅 메시지와 영화 정보 간의 코사인 유사도 계산
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # 유사도가 높은 영화 추천
        chat_similarity_scores = similarity_matrix[:-len(movie_info), -len(movie_info):]
        top_similar_indices = chat_similarity_scores.argmax(axis=1)

        recommended_movies[member_id] = [movie_info[idx] for idx in top_similar_indices]

        # 중복 영화 제거
        recommended_movies[member_id] = remove_duplicate_movies(recommended_movies[member_id])

    return recommended_movies

if __name__ == "__main__":
    item_data = get_item_data()
    chat_data = get_chat_data()

    '''
    recommended_movies = recommend_movies_for_members(item_data, chat_data)

    for member_id, movies in recommended_movies.items():
        print(f"{member_id} 멤버에게 추천하는 영화:")
        for movie in movies:
            print(movie)
    '''
