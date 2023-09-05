import mysql.connector
from get_data import get_view

# MySQL 데이터베이스 연결 설정
db_config = {
    "host": "127.0.0.1",  # 호스트 주소
    "user": "root",       # 사용자 이름
    "password": "hanul",  # 비밀번호
    "database": "hanuldb",  # 데이터베이스 이름
    "port": 3306          # MySQL 포트 번호
}

# 감정과 장르를 구분하는 함수
def distinguish(item):
    # 감정 키워드 목록
    emotions = ['분노', '슬픔', '기쁨', '걱정', '불안', '설렘', '우울', '공포']
    
    # 감정인 경우
    if item in emotions:
        return 'emotion'
    # 장르인 경우
    else:
        return 'genre'

# 미니 챗 영화 추천 - 감정
def emotion_minichatmovie(item):
    e_view = ""
    
    if item == '분노':
        e_view = 'anger'
    elif item == '슬픔':
        e_view = 'sad'
    elif item == '기쁨':
        e_view = 'joy'
    elif item == '걱정':
        e_view = 'worry'
    elif item == '불안':
        e_view = 'anxiety'
    elif item == '설렘':
        e_view = 'neutral'
    elif item == '우울':
        e_view = 'depression'
    elif item == '공포':
        e_view = 'fear'

    print(e_view)
    recommended_movies = get_view(e_view)
    print(recommended_movies)

    return recommended_movies

# 미니 챗 영화 추천 - 장르
def genre_minichatmovie(genres):
    # 한국어 장르를 영어 장르로 변환하는 사전
    genre_mapping = {
        '드라마': 'Drama',
        '로맨스': 'Romance',
        '가족': 'Family',
        '액션': 'Action',
        '범죄': 'Crime',
        '음악': 'Music',
        '코미디': 'Comedy',
        '판타지': 'Fantasy',
        '모험': 'Adventure',
        '애니메이션': 'Animation'
    }

    # 한국어 장르를 영어 장르로 변환
    english_genres = [genre_mapping[genre] for genre in genres if genre in genre_mapping]

    # 각 장르에 해당하는 아이템을 데이터베이스에서 가져오는 로직
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    genre_placeholders = ', '.join(['%s'] * len(english_genres))
    
    if english_genres:
        query = f"SELECT * FROM item WHERE genre_name IN ({genre_placeholders})"
        cursor.execute(query.encode('utf8'), tuple(english_genres))
        recommended_movies = cursor.fetchall()
    else:
        recommended_movies = []

    return recommended_movies
