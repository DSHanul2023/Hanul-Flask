import sys
sys.path.append(r'')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt, Mecab
import time
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
# okt = Okt()

tagger = Mecab(r'C:\mecab\share\mecab-ko-dic')

# 텍스트 전처리 및 토큰화 함수
def preprocess_text(text):
    words = tagger.nouns(text)
    return ' '.join(words)
# 영화 정보 전처리 함수
def preprocess_movie_info(movie_info):
    preprocessed_movie_info = [preprocess_text(f"{info['title']} {info['description']} {info['genre']}") for info in movie_info]
    return preprocessed_movie_info

# "item" 테이블 데이터 가져오기
def get_item():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    query = "SELECT * FROM item"
    cursor.execute(query)

    item_data = cursor.fetchall()

    cursor.close()
    connection.close()

    return item_data

# "item" 테이블에서 뷰 데이터 가져오기
def get_view(view):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    query = f"SELECT * FROM `{view}`"
    cursor.execute(query)

    item_data = cursor.fetchall()

    cursor.close()
    connection.close()


def preprocess_movie_info(movie_info):
    preprocessed_movie_info = [preprocess_text(f"{info['title']} {info['description']} {info['genre']}") for info in movie_info]
    return preprocessed_movie_info



# "chat" 테이블 데이터 가져오기
def get_chat(member_id):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    query = "SELECT * FROM chat WHERE member_id = %s"
    cursor.execute(query, (member_id,))  # 매개변수를 통해 SQL 쿼리 파라미터 전달

    chat_data = cursor.fetchall()

    cursor.close()
    connection.close()

    preprocessed_chat_data = []

    for chat in chat_data:
        preprocessed_chat_data.append(chat[1])

    return preprocessed_chat_data

