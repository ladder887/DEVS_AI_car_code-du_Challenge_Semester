import socket
import time
import json

def collect_data():
    data = {"dict" : 15, "speed" : 40, "track" : 4, "object_name" : "red"}
    return data

def send_data(data):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('192.168.1.134', 9999)
    sock.connect(server_address)
    json_data = json.dumps(data)
    
    sock.sendall(json_data.encode())

    sock.close()

while True:
    data = collect_data()
    send_data(data)
    print(data)
    time.sleep(2)