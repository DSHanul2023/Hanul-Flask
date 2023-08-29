from emotion import predict
from get_data import get_chat

# 추천 영화 중 중복 제거
def remove_duplicate_movies(movies):
    unique_movies = []
    movie_ids_seen = set()

    for movie in movies:
        if movie['movie_id'] not in movie_ids_seen:
            unique_movies.append(movie)
            movie_ids_seen.add(movie['movie_id'])

    return unique_movies

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

