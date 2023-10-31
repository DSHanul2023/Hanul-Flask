from justwatch import justwatchapi, JustWatch
from requests.exceptions import HTTPError
import json
import requests
from bs4 import BeautifulSoup

def get_provider_data(tmdb_id, tmdb_name):
    # print('tmdb_id : ', tmdb_id)
    just_watch = JustWatch(country='KR')
    titles = just_watch.search_for_item(query=tmdb_name)
    results= {"tmdb_id": tmdb_id,
              "buy": [],
              "flatrate":[],
              "rent":[]}

    tmdb_id = int(tmdb_id)

    # 중복을 제거한 결과를 저장할 리스트를 생성합니다.
    unique_data_list = []
    # 이미 처리한 provider_id와 monetization_type 조합을 저장할 집합(set)을 생성합니다.
    processed_combinations = set()
    for title in titles['items']:
        data_tmdb_id = None
        for score in title['scoring']:
            if score['provider_type'] == 'tmdb:id':
                data_tmdb_id = score['value']
                break

        if (tmdb_id == data_tmdb_id):
            for item in title['offers']:
                provider_id = item["provider_id"]
                monetization_type = item["monetization_type"]

                combination = (provider_id, monetization_type)
                if combination in processed_combinations:
                    continue

                processed_combinations.add(combination)
                unique_data_list.append(item)

            # 중복이 없는 데이터 리스트를 출력합니다.
            for unique_item in unique_data_list:
                print(unique_item)
            break

    providers_data = just_watch.get_providers()

    for unique_item in unique_data_list:
        result = {
            "provider_name": "",
            "url": ""
        }
        for provider in providers_data:
            if provider["id"] == unique_item['provider_id']:
                # print(unique_item['provider_id'])
                # print(provider["id"])
                result["provider_name"] = provider['clear_name']
                break

        result["url"] = unique_item['urls']['standard_web']
        if(unique_item['monetization_type'] == 'buy'):
            results['buy'].append(result)
        elif(unique_item['monetization_type'] == 'flatrate'):
            results['flatrate'].append(result)
        elif(unique_item['monetization_type'] == 'rent'):
            results['rent'].append(result)

    return results

def get_provider_crawling (tmdb_id, url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'}
    # requests를 사용하여 웹 페이지의 HTML 데이터 가져오기
    response = requests.get(url, headers=headers)

    results = {"tmdb_id": tmdb_id,
               "buy": [],
               "flatrate": [],
               "rent": []}

    if response.status_code == 200:
        # Beautiful Soup을 사용하여 HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        # li 태그 중 클래스 이름이 "ott_filter_best_price"인 항목 찾기
        ott_items = soup.find_all('li', class_='ott_filter_best_price')

        # 각 ott_item에서 a 태그의 title 값을 파싱하여 provider_type, title, provider_name을 추출
        for item in ott_items:
            a_tag = item.find('a')
            result = {
                "provider_name": "",
                "url": ""
            }
            if a_tag:
                title = a_tag['title']
                href = a_tag['href']

                parts = title.split(' on ')
                provider_name = parts[1]
                result['provider_name'] = provider_name
                result['url'] = href

                provider_type = parts[0].split()[0]
                # print(provider_type)
                if provider_type == 'Rent':
                    results['rent'].append(result)
                elif provider_type == 'Buy':
                    results['buy'].append(result)
                elif provider_type == 'Watch':
                    results['flatrate'].append(result)

        print(results)
        return results
    else:
        print('Failed to retrieve the web page. Status code:', response.status_code)
        return None


if __name__ == '__main__':
    # data = get_provider_data(11887, "하이 스쿨 뮤지컬: 졸업반")
    # print(data)
    # just_watch = JustWatch()
    #
    # results = just_watch.search_for_item(query='the matrix')
    get_provider_crawling(605, 'https://www.themoviedb.org/movie/605-the-matrix-revolutions/watch?locale=KR')
