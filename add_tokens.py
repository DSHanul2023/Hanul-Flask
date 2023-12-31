import sys
# sys.path.append(r'C:\Welover\Flask-hanul\venvs\venv\Lib\site-packages')
# sys.path.append(r'C:\Welover\Flask-hanul\venv\Lib\site-packages') #SCE
sys.path.append('/home/ubuntu/Flask-hanul/venv/lib/python3.10/site-packages')
from konlpy.tag import Mecab
import mysql.connector

db_config = {
    "host": "database-2.cmcn6ounorqr.ap-northeast-2.rds.amazonaws.com",  # 호스트 주소
    "user": "hanul_admin",       # 사용자 이름
    "password": "hanulhanul",  # 비밀번호
    "database": "hanuldb",  # 데이터베이스 이름
    "port": 3306,         # MySQL 포트 번호
    "charset": "utf8"     # 문자셋 지정
}

# Mecab 토크나이저 생성
# tagger = Mecab(r'C:\mecab\share\mecab-ko-dic')
tagger = Mecab(dicpath="/home/ubuntu/Flask-hanul/mecab/mecab-ko-dic-2.1.1-20180720")

# Mecab 토큰화 함수
def mecab_preprocess(text):
    words = tagger.nouns(text)
    return ' '.join(words)

# 1. 처음 한 번 실행
if __name__ == "__main__":
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 'item' 테이블에서 'title'과 'detail' 컬럼 가져오기
    query = "SELECT item_nm, item_detail FROM item"
    cursor.execute(query.encode('utf8'))
    results = cursor.fetchall()

    # add_tokens.py 처음에 주석 후 실행 -> 주석 제거
    # query = "ALTER TABLE item DROP COLUMN tokens;"
    # cursor.execute(query.encode('utf8'))

    query = "ALTER TABLE item ADD tokens LONGTEXT"
    cursor.execute(query.encode('utf8'))

    # 토큰화 및 'token' 열 생성
    for title, detail in results:
        if(detail !=""):
            nouns = mecab_preprocess(detail)
        else:
            nouns ="" # details 컬럼 값이 null이면 토큰 x
        update_query = f"UPDATE item SET tokens = %s WHERE item_nm = %s"
        cursor.execute(update_query.encode('utf8'), (nouns, title))

    # 변경 내용 커밋 및 연결 닫기
    conn.commit()
    cursor.close()
    conn.close()
