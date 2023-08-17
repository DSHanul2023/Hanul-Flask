from flask import Flask, request, jsonify
import json
import os
import numpy as np
import torch
from model.kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer

app = Flask(__name__)

# KoGPT-2 모델 초기화 및 로드
root_path = '.'
checkpoint_path = f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

checkpoint = torch.load(save_ckpt_path, map_location=device)

model = DialogKoGPT2()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = get_kogpt2_tokenizer()

# Flask 모델 로직을 사용하여 데이터 처리
def process_with_model(question):
    tokenized_indexs = tokenizer.encode(question)
    input_ids = torch.tensor([tokenizer.bos_token_id,] + tokenized_indexs + [tokenizer.eos_token_id]).unsqueeze(0)
    sample_output = model.generate(input_ids=input_ids)

    answer = tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:], skip_special_tokens=True)

    # 두 번째 마침표까지만 저장
    second_dot_index = answer.find('.', answer.find('.') + 1)
    if second_dot_index != -1:
        answer = answer[:second_dot_index + 1]


    return answer

@app.route('/process', methods=['POST'])
def process_data():
    request_data = request.json
    question = request_data.get('question', '')  # 'name' 필드에서 데이터를 가져옴
    answer = process_with_model(question)  # 모델로 응답 생성

    response_data = {
        "answer": answer
    }

    return response_data

if __name__ == '__main__':
    app.run()
