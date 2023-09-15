from justwatch import justwatchapi, JustWatch
from requests.exceptions import HTTPError
import json

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

if __name__ == '__main__':
    data = get_provider_data(11887, "하이 스쿨 뮤지컬: 졸업반")
    print(data)