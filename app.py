from flask import Flask, request, jsonify
import json
app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process_data():
    request_data = request.json
    # Flask 모델을 사용하여 데이터 처리
    result = process_with_model(request_data)

    # Spring Boot에 전달할 응답을 준비
    response_data = {
        "result": result
    }
    # JSON 형식의 문자열로 변환
    response_data = json.dumps(result)

    # 백슬래시 제거 후 다시 JSON 형식의 문자열로 변환
    result_json_without_backslashes = json.loads(response_data)
    result_json_without_backslashes = json.dumps(result_json_without_backslashes, ensure_ascii=False)

    return result_json_without_backslashes


def process_with_model(data):
    # Flask 모델 로직을 사용하여 데이터 처리
    # ...
    return data


if __name__ == '__main__':
    app.run()