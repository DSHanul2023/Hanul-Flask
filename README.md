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

- 가상환경 아래 scripts 폴더에
[requirements.txt](https://github.com/DSHanul2023/Flask-hanul/files/12460712/requirements.txt)
[Uploading reqblinker==1.6.2
certifi==2023.7.22
charset-normalizer==3.2.0
click==8.1.7
colorama==0.4.6
filelock==3.12.3
Flask==2.3.3
fsspec==2023.6.0
gluonnlp==0.8.0
graphviz==0.20.1
huggingface-hub==0.16.4
idna==3.4
install==1.3.5
itsdangerous==2.1.2
Jinja2==3.1.2
joblib==1.3.2
JPype1==1.4.1
kobert-tokenizer @ git+https://github.com/SKTBrain/KoBERT.git@47a69af87928fc24e20f571fe10c3cc9dd9af9a3#subdirectory=kobert_hf
kogpt2-transformers==0.4.0
konlpy==0.6.0
lxml==4.9.3
MarkupSafe==2.1.3
mecab-ko-dic-msvc==0.999
mecab-ko-msvc==0.999
mpmath==1.3.0
mxnet==1.8.0
mysql-connector-python==8.1.0
networkx==3.1
nltk==3.8.1
numpy==1.23.5
packaging==23.1
protobuf==4.21.12
PyYAML==6.0.1
regex==2023.8.8
requests==2.31.0
safetensors==0.3.3
scikit-learn==1.3.0
scipy==1.11.2
sympy==1.12
threadpoolctl==3.2.0
tokenizers==0.13.3
torch==2.0.1
tqdm==4.66.1
transformers==4.32.1
typing_extensions==4.7.1
urllib3==2.0.4
Werkzeug==2.3.7
uirements.txt…]()
