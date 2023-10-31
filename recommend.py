from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(r'C:\Welover\Flask-hanul\venvs\venv\Lib\site-packages')
from emotion import predict
from operator import itemgetter
from get_data import get_chat, get_view, get_saved,preprocess_text
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

    query = "CREATE VIEW anger AS SELECT * FROM item WHERE genre_name LIKE '%%action%' AND genre_name LIKE '%Crime%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW worry AS SELECT * FROM item WHERE genre_name LIKE '%Drama%' AND genre_name LIKE '%Romance%' AND genre_name LIKE '%Family%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW anxiety AS SELECT * FROM item WHERE genre_name LIKE '%Romance%' AND genre_name LIKE '%Drama%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW depression AS SELECT * FROM item WHERE genre_name LIKE '%Drama%' AND genre_name LIKE '%Music%' AND genre_name LIKE '%Comedy%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW fear AS SELECT * FROM item WHERE genre_name LIKE '%Animation%' AND genre_name LIKE '%%Family%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW sad AS SELECT * FROM item WHERE genre_name LIKE '%Drama%' AND genre_name LIKE '%Animation%' AND genre_name LIKE '%Comedy%'"
    cursor.execute(query.encode('utf8'))

    query = "CREATE VIEW joy AS SELECT * FROM item WHERE genre_name LIKE '%%Fantasy%' AND genre_name LIKE '%Adventure%' AND genre_name LIKE '%action%'"
    cursor.execute(query.encode('utf8'))
    
    query = "CREATE VIEW neutral AS SELECT * FROM item WHERE genre_name LIKE '%Adventure%'"
    cursor.execute(query.encode('utf8'))


# 사용자 발화와 영화간 유사도 계산(줄거리) / 딕셔너리 반환
def descr_based_recommender(item_data, chat_data):
    # 영화 정보 데이터
    # movie_info = [{'genre': item[3], 'title': item[5], 'movie_id': item[0], 'tokens': item[9]} for item in item_data]
    movie_info = [{'genre': item[3], 'title': item[5], 'movie_id': item[0], 'tokens': item[8]} for item in item_data]

    print("movie_info 길이:", len(movie_info))
    for movie in movie_info:
        print(movie)

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
    cosine_similarities = cosine_similarity(vectorizer.transform(user_says_list), vectorized_data).flatten()
    
    tokens.pop()
    
    for idx, sim in enumerate(cosine_similarities):
        if idx < len(movie_info):
            movie_info[idx]['dbr_cosine_similarity'] = sim
        else:
            print(f"Index {idx} is out of range for movie_info list.")
    
    return movie_info


# 영화간 유사도 계산(메타 데이터) - ver1
# 사용자 선호 영화 메타 데이터를 하나의 문자열로 합친 뒤 유사도 계산
def md_based_recommender(item_data, saved_data):
    # 영화 정보 데이터
    movie_info = [{'id': item[0], 'director': item[2], 'crew': item[1], 'genre': item[3], 'keywords': item[6]} for item in item_data]
    saved_info = [{'id': item[0], 'director': item[2], 'crew': item[1], 'genre': item[3], 'keywords': item[6]} for item in saved_data]

    # movie_info 각 데이터 문자열 전처리 및 인덱스 추가
    
    for idx, movie in enumerate(movie_info):
        if isinstance(movie['director'], str):
            movie['director'] = ''
        else:
            movie['director'] = str.lower(movie['director']).replace(" ", "")
            movie['crew'] = str.lower(movie['crew']).replace(" ", "").replace(",", " ")
            movie['genre'] = str.lower(movie['genre']).replace(" ", "").replace(",", " ")
            movie['keywords'] = str.lower(movie['keywords']).replace(" ", "").replace(",", " ")
            movie['idx'] = idx
    soup = []

    for movie in movie_info:
        movie['soup'] = ''.join(movie['keywords']) + '' + ''.join(movie['crew']) + '' + movie['director'] + '' + ''.join(movie['genre'])
        soup.append(movie['soup'])

    # saved_info 각 데이터 문자열 전처리
    for idx, movie in enumerate(saved_info):
        if isinstance(movie['director'], str):
            movie['director'] = ''
        else:
            movie['director'] = str.lower(movie['director']).replace(" ", "")
            movie['crew'] = str.lower(movie['crew']).replace(" ", "").replace(",", " ")
            movie['genre'] = str.lower(movie['genre']).replace(" ", "").replace(",", " ")
            movie['keywords'] = str.lower(movie['keywords']).replace(" ", "").replace(",", " ")

    saved_soup = ''

    # saved_info 데이터 통합
    for movie in saved_info:
        temp = ''.join(movie['keywords']) + '' + ''.join(movie['crew']) + '' + movie['director'] + '' + ''.join(movie['genre'])
        saved_soup = saved_soup + ' ' + temp

    soup.append(saved_soup)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(soup)

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    sim_scores = list(enumerate(cosine_sim[len(cosine_sim)-1]))
    sim_scores.pop()

    return sim_scores, 1


# 영화간 유사도 계산(메타 데이터) - ver2
# 사용자 선호 영화 개별 데이터와 유사도 계산 후 합산
def md_based_recommender2(item_data, saved_data):
    # 영화 정보 데이터
    movie_info = [{'id': item[0], 'director': item[2], 'crew': item[1], 'genre': item[3], 'keywords': item[6]} for item in item_data]
    saved_info = [{'id': item[0], 'director': item[2], 'crew': item[1], 'genre': item[3], 'keywords': item[6]} for item in saved_data]

    # movie_info 각 데이터 문자열 전처리 및 인덱스 추가
    
    for idx, movie in enumerate(movie_info):
        if isinstance(movie['director'], str):
            movie['director'] = ''
            movie['crew'] = str.lower(movie['crew']).replace(" ", "").replace(",", " ")
            movie['genre'] = str.lower(movie['genre']).replace(" ", "").replace(",", " ")
            movie['keywords'] = str.lower(movie['keywords']).replace(" ", "").replace(",", " ")
            movie['idx'] = idx
            movie['score'] = 0
        else:
            movie['director'] = str.lower(movie['director']).replace(" ", "")
            movie['crew'] = str.lower(movie['crew']).replace(" ", "").replace(",", " ")
            movie['genre'] = str.lower(movie['genre']).replace(" ", "").replace(",", " ")
            movie['keywords'] = str.lower(movie['keywords']).replace(" ", "").replace(",", " ")
            movie['idx'] = idx
            movie['score'] = 0
    soup = []

    for movie in movie_info:
        movie['soup'] = ''.join(movie['keywords']) + '' + ''.join(movie['crew']) + '' + movie['director'] + '' + ''.join(movie['genre'])
        soup.append(movie['soup'])

    # saved_info 각 데이터 문자열 전처리 및 인덱스 추가
    for idx, movie in enumerate(saved_info):
        if isinstance(movie['director'], str):
            movie['director'] = ''
            movie['crew'] = str.lower(movie['crew']).replace(" ", "").replace(",", " ")
            movie['genre'] = str.lower(movie['genre']).replace(" ", "").replace(",", " ")
            movie['keywords'] = str.lower(movie['keywords']).replace(" ", "").replace(",", " ")
        else:
            movie['director'] = str.lower(movie['director']).replace(" ", "")
            movie['crew'] = str.lower(movie['crew']).replace(" ", "").replace(",", " ")
            movie['genre'] = str.lower(movie['genre']).replace(" ", "").replace(",", " ")
            movie['keywords'] = str.lower(movie['keywords']).replace(" ", "").replace(",", " ")
    saved_soup = []

    for movie in saved_info:
        movie['soup'] = ''.join(movie['keywords']) + '' + ''.join(movie['crew']) + '' + movie['director'] + '' + ''.join(movie['genre'])
        saved_soup.append(movie['soup'])

    size = len(saved_soup)

    count = CountVectorizer(stop_words='english')
    for ssoup in saved_soup:
        soup.append(ssoup)
        count_matrix = count.fit_transform(soup)

        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        sim_scores = list(enumerate(cosine_sim[len(cosine_sim)-1]))
        sim_scores.pop()
        soup.pop()

        for movie, sim_score in zip(movie_info, sim_scores):
            movie['score'] = movie['score'] + sim_score[1]

    # enumerate를 사용하여 'score' 값을 추출하고 유사도 리스트 생성
    fsim_scores = [(idx, movie['score']) for idx, movie in enumerate(movie_info)]

    return fsim_scores, size

def get_emotion(user_id):

    print("get_emotion3")
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

    return e_view

def recommendation(user_id, saved):
    chat_data = get_chat(user_id)
    predicted_emotions = []
    user_says = []

    # 사용자 발화 감정 분석
    for text in chat_data:
        predicted_emotions.append(predict(text))
        user_says.append(preprocess_text(text))

    print("get_saved(saved) : ", get_saved(saved))

    if(len(get_saved(saved)) == 0):
        saved_data = 0
    else:
        saved_data = get_saved(saved)[0]

    e_view = get_emotion(user_id)

    # 분류된 감정에 맞는 view에서 아이템 불러오기
    item_data = get_view(e_view) # 튜플 반환
    # label = ['id', 'cast', 'director', 'genre', 'descr', 'title', 'keyword', 'url', 'mem', 'token']
    label = ['id', 'cast', 'director', 'genre', 'descr', 'title', 'keyword', 'url', 'token']
    item_dic = []

    for item in item_data:
        item_dic.append(dict(zip(label, item)))

    # 줄거리 기반 유사도 계산    
    recommend_movies = descr_based_recommender(item_data, user_says)

    # 줄거리 기반 추천 유사도 저장
    for idx, movie in enumerate(recommend_movies):
        if(item_dic[idx]['title']!=movie['title']):
            print('index out of range')
            break
        item_dic[idx]['dbr_cosine_similarity'] = movie['dbr_cosine_similarity']

    # 북마크 유사도 계산
    # sim_scores, size = md_based_recommender(item_data, saved_data)

    # 북마크 유사도 계산2
    if saved_data != 0:
        sim_scores, size = md_based_recommender2(item_data, saved_data)
        # print(sim_scores)
        
        for idx, score in enumerate(sim_scores):
            item_dic[idx]['md_cosine_similarity'] = score[1]

        for item in item_dic:
            value = item['md_cosine_similarity'] + item['dbr_cosine_similarity']
            min_value = 0
            max_value = size
            target_min = 0
            target_max = 10
            item['score'] = (value - min_value) / (max_value - min_value) * (target_max - target_min) + target_min

    else:
        for item in item_dic:
            value = item['dbr_cosine_similarity']
            min_value = 0
            max_value = 1
            target_min = 0
            target_max = 10
            item['score'] = (value - min_value) / (max_value - min_value) * (target_max - target_min) + target_min

    sort = sorted(item_dic, key=itemgetter('score'), reverse=True)
    recommended_movies = sort[0:8]
    for movie in recommended_movies:
        print('제목: ',{movie['title']},'\n키워드: ',{movie['keyword']},'\n유사도: ',{movie['score']},'\n\n')

    return recommended_movies

if __name__ == "__main__":
    recommendation('a12334')

'''

def recommend_movies_for_members(pre_item_data, item_data, chat_data):
    # 영화 정보 데이터
    movie_info = [{'item_id': item[0], 'genre': item[1], 'description': item[2], 'title': item[3], 'movie_id': item[4], 'image_url': item[5], 'member_id': item[6], 'tokens': item[7]} for item in item_data]

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
'''

if __name__ == "__main__":
    create_view()
    # recommendation('402899838a23ffb6018a2400f47504f6')

