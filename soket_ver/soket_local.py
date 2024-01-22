import socket
import json
import pandas as pd
import os

server_address = ('', 9999)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(server_address)
sock.listen(1)

while True:
    connection, client_address = sock.accept()
    json_data = connection.recv(1024).decode()

    data = json.loads(json_data)
    df = pd.DataFrame(data, index = [0])
    print(df)
    saved = "C:\\Users\\User\\Desktop\\AI_Car_CSV\\Car_data.csv"
    if not os.path.exists(saved):
        df.to_csv(saved, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(saved, mode='a', encoding='utf-8-sig', header=False)

    connection.close()