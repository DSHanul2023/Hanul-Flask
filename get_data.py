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
def get_chat_data():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    query = "SELECT * FROM chat"
    cursor.execute(query)

    chat_data = cursor.fetchall()

    cursor.close()
    connection.close()

    return chat_data

def recommend_movies_for_members(item_data, chat_data):
    # 영화 상세 설명 데이터
    movie_descriptions = [item[1] for item in item_data]
    
    # 고유한 멤버 ID 가져오기
    member_ids = set(chat[0] for chat in chat_data)

    # 각 멤버의 채팅 메시지를 저장할 딕셔너리
    member_chat_messages = {}

    # 채팅 데이터를 멤버별로 그룹화하여 처리
    for chat in chat_data:
        member_id, message = chat[0], chat[1]
        if member_id not in member_chat_messages:
            member_chat_messages[member_id] = []
        member_chat_messages[member_id].append(message)

    # 멤버별 추천 영화를 저장하는 딕셔너리
    recommended_movies = {}

    # 멤버별 채팅 데이터를 가져와서 처리
    for member_id in member_ids:
        chat_messages = member_chat_messages.get(member_id, [])

        # 데이터 전처리
        preprocessed_chat_messages = [preprocess_text(text) for text in chat_messages]
        preprocessed_movie_descriptions = [preprocess_text(text) for text in movie_descriptions]

        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_chat_messages + preprocessed_movie_descriptions)

        # 채팅 메시지와 영화 설명 간의 코사인 유사도 계산
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # 유사도가 높은 영화 추천
        chat_similarity_scores = similarity_matrix[:-len(movie_descriptions), -len(movie_descriptions):]
        top_similar_indices = chat_similarity_scores.argmax(axis=1)

        recommended_movies[member_id] = [item_data[idx][1] for idx in top_similar_indices]

        return recommended_movies

if __name__ == "__main__":
    item_data = get_item_data()
    chat_data = get_chat_data()

    recommended_movies = recommend_movies_for_members(item_data, chat_data)

    for member_id, movies in recommended_movies.items():
        print(f"{member_id} 멤버에게 추천하는 영화:")
        for movie in movies:
            print(movie)