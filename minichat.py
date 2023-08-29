import mysql.connector
# MySQL 데이터베이스 연결 설정
db_config = {
    "host": "127.0.0.1",  # 호스트 주소
    "user": "root",       # 사용자 이름
    "password": "hanul",  # 비밀번호
    "database": "hanuldb",  # 데이터베이스 이름
    "port": 3306          # MySQL 포트 번호
}
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
    if selected_genres:
        query = f"SELECT * FROM item WHERE genre_name IN ({genre_placeholders})"
        cursor.execute(query.encode('utf8'), tuple(selected_genres))
        recommended_movies = cursor.fetchall()
    else:
        recommended_movies = []

    cursor.close()
    conn.close()

    return recommended_movies