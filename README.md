We:Lover - 심리 상담 AI 챗봇 기반 이야기 치료법 제공 서비스
---
### 📍 팀원 소개
<table>
  <tr>
    <td align="center"><a href="https://github.com/pjho4746">박지호(PM)</a></td>
    <td align="center"><a href="https://github.com/seoyeong12">서채은</td>
    <td align="center"><a href="https://github.com/chaeeunnn">김서영</td>
    <td align="center"><a href="https://github.com/jayy118">이재현</td>
  </tr>
  <tr>
    <td align="center">pjho4746@gmail.com</td>
    <td align="center">jytp922@gmail.com</td>
    <td align="center">qlqltjdud@gmail.com</td>
    <td align="center">hyun918@duksung.ac.kr</td>
  </tr>
  <tr>
    <td align="center">풀스택 개발자</td>
    <td align="center">풀스택 개발자</td>
    <td align="center">풀스택 개발자</td>
    <td align="center">풀스택 개발자</td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/DSHanul2023/Hanul-Backend/assets/126854628/606bbf95-fe28-4e01-8789-e8ba93e05995?raw=true" width="150px"></td>
    <td align="center"><img src="https://github.com/DSHanul2023/Hanul-Backend/assets/126854628/e7a35f81-41e5-49d0-b25a-289d064dd551?raw=true" width="150px"></td>
    <td align="center"><img src="https://github.com/DSHanul2023/Hanul-Backend/assets/126854628/38607519-a067-4d00-9e9a-fe44fac12fad?raw=true" width="150px"></td>
    <td align="center"><img src="https://github.com/DSHanul2023/Hanul-Backend/assets/126854628/6a9a0a7d-4678-4558-b8db-565e6b9c9869?raw=true" width="150px"></td>
  </tr>
</table>

---

### 📍 개발 배경
**높은 정신질환 유병률과 낮은 정신건강 서비스 이용률**<br>
정신건강은 현대 사회에서 중요한 문제 중 하나로 부상하고 있음. 국내 정신질환 평생 유병률은 27.8%로 나타나고 있으며, 이에 따른 정신건강 서비스의 필요성이 커지고 있음. 그러나 국내 정신건강 서비스 이용률은 7.2%에 그침. 캐나다나 미국의 경우 정신건강 서비스 이용률이 각각 43.1%, 46.5%에 이른다는 점을 고려해 보았을 때 우리나라는 현저하게 낮은 수준임.<br><br>
**정신건강 서비스 이용률 증대**<br>
글로벌 디지털 정신건강 관리 솔루션 시장은 코로나 이후 크게 성장할 것으로 예측됨. 정신건강 앱 다운로드 수가 코로나 발생 이전 대비 17.6% 증가했다는 보고가 있음. 심리적 불안 지속과 이동성 제약에 따라, 디지털 기술 기반의 비대면 정신건강 관리 서비스에 대한 소비자 인식과 수용도가 개선되었다고 봄. 

---

### 📍 기획 의도
웹 기반 챗봇 상담 서비스로써 국내 정신건강 서비스 이용률이 저조한 주요 요인인 비용 문제, 시공간의 제약, 사회의 부정적 인식, 전문기관에 대한 정보 부족에 따른 심리적 부담감을 해소하고자 함. 고민 상담 내용을 기반으로 영화 추천 및 위로와 조언을 제공하는 웹 기반 서비스를 구현하였고 사용자가 심리적 거리를 줄이고 쉽게 접근할 수 있도록 애완동물 콘셉트를 활용함.<br><br>

AI 챗봇을 활용한 고민 상담을 통해 위로 및 제안을 제공하고, 대화 내용을 기반으로 영화를 추천하는 시스템을 제안. KoBert 모델을 이용하여 사용자의 감성을 분석하고, KoGPT 모델을 활용해 챗봇 응답을 생성.

---
### 📍 아키텍처
![아키텍처](https://github.com/DSHanul2023/Hanul-Backend/assets/126854628/59080cd0-ae92-446d-bea6-c119fb041ec0)


---
### 📍 메인 기능
**1. 챗봇 서비스**<br>
- Dialogflow Webhook을 통해 사용자 입력 처리 및 전체 대화 흐름을 제어
- KoGPT2 모델로 실제 상담 데이터를 미리 학습한 후 사용자 입력에 대한 응답 생성

 **2. 감정 분류**<br>
- KoBert 모델을 활용하여 사용자 발화에 대한 감정 분석 수행
- 상담 데이터셋과 단발성 대화 데이터셋 활용하여 학습
- 총 8가지 감정으로 분류하고 각 감정에 미리 정의된 정르로 매핑

**3. 맞춤 영화 추천**<br>
- TMDB API를 사용하여 영화 데이터셋 구축
- 감정-장르 매핑으로 가장 많이 나온  장르로 영화 1차 필터링
- 사용자 발화와 영화 줄거리 간의 유사도 계산
- 사용자 북마크 영화와 1차 필터링된 영화와 유사도 계산
- 두 개의 유사도를 더 해 결과값이 높은 상위 8개의 추천 영화 제공

**4. 영화 추천 체험**<br>
- 간단한 설문조사를 통해 영화 추천 기능 체험 가능
- 사용자의 현재 감정과 선호하는 장르를 선택하여 영화 추천을 받을 수 있음

---


### 📍 시연 영상
최종 결과물  
https://drive.google.com/file/d/169lrsR1uvXOvaUwd7PUo9AbA7ArDlu7L/view?usp=sharing  <br><br>
중간 결과물 - 자막 설명 O  
https://drive.google.com/file/d/1O8J9JzyTLuCX4OO8brG8q5OUO2SSy6fR/view?usp=sharing 

---
### 📍 활동
**고민 상담 챗봇 기반 영화 추천 시스템** <br>
*투고 날짜 : 2023.09.20* <br>
*논문 번호 : KIPS_C2023B0335* <br><br>

**한국정보산업연합회** <br>
*이브와 ICT 멘토링* <br>

---
### 가상환경 패키지 설치

- pip install flask torch kogpt2-transformers mysql-connector-python
- pip install nltk scikit-learn konlpy
- pip install gluonnlp==0.8.0
- pip install tqdm pandas sentencepiece transformers
- pip install numpy==1.23.5
- pip install mxnet -f https://dist.mxnet.io/python/cpu
- pip install "git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf"

### venv/lib/site-packages에 모듈 설치 확인됐는데 No module found 나올 때
- app.py 맨 위에 작성
- import sys
- sys.path.append(r'site-packages 경로')

- 가상환경 아래 scripts 폴더에
[requirements.txt](https://github.com/DSHanul2023/Flask-hanul/files/12460712/requirements.txt)
- pip install -r requirements.txt

### 실행 순서
- python add_tokens.py
```bash
query = "ALTER TABLE item DROP COLUMN tokens;"
cursor.execute(query.encode('utf8'))
# 처음 생성할 때는 이 부분 주석 처리
```
- python app.py
```bash
create_view()
# 처음 실행할 때만 실행
# 생성 후에는 주석 처리
```
