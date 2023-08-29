# Flask-hanul

가상환경 패키지 설치

- pip install flask torch kogpt2-transformers mysql-connector-python
- pip install nltk scikit-learn konlpy
- pip install gluonnlp==0.8.0
- pip install tqdm pandas sentencepiece transformers
- pip install numpy==1.23.5
- pip install mxnet -f https://dist.mxnet.io/python/cpu
- pip install "git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf"

## venv/lib/site-packages에 모듈 설치 확인됐는데 No module found 나올 때
- app.py 맨 위에 작성
- import sys
- sys.path.append(r'site-packages 경로')
