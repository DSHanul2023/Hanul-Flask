import mysql.connector

# MySQL 데이터베이스 연결 설정
db_config = {
    "host": "127.0.0.1",  # 호스트 주소
    "user": "root",       # 사용자 이름
    "password": "hanul",  # 비밀번호
    "database": "hanuldb",  # 데이터베이스 이름
    "port": 3306          # MySQL 포트 번호
}


# "item" 테이블 데이터 가져오기
def get_item_data():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    query = "SELECT * FROM item"
    cursor.execute(query)

    item_data = cursor.fetchall()

    cursor.close()
    connection.close()

    return item_data

# "chat" 테이블 데이터 가져오기
def get_chat_data():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    query = "SELECT * FROM chat"
    cursor.execute(query)

    chat_data = cursor.fetchall()

    cursor.close()
    connection.close()

    return chat_data

if __name__ == "__main__":
    item_data = get_item_data()
    chat_data = get_chat_data()

    print("Item Data:")
    for item in item_data:
        print(item)

    print("\nChat Data:")
    for chat in chat_data:
        print(chat)
