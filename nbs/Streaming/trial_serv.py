import socket
import os
from _thread import *
import datetime
import time

ServerSocket = socket.socket()  # Create a socket
host = "127.0.0.1"  # Define host
port = 10000  # Define port
global ThreadCount
ThreadCount = 0  # Define thread count
try:
    ServerSocket.bind((host, port))  # Bind port and host
except socket.error as e:
    print(str(e))

print("Waitiing for a Connection..")
ServerSocket.listen(5)  # Listen to 5 connection before refusing new connection

global data


class SocketClass:
    def __init__(self):
        return

    def createSocket(self):
        Client, address = ServerSocket.accept()
        times = datetime.datetime.now()
        print("\nTime: " + str(times))
        print("Connected to: " + address[0] + ":" + str(address[1]))
        print("\n")
        return Client

    def threaded_client(self, client1, client2):
        # client1.send(str.encode('Welcome to the Server....Client\n'))
        client2.send(str.encode("Welcome to the Server....Client2\n"))
        while True:
            data = client1.recv(2048)
            print("Client says: " + data.decode("utf-8"))
            client2.send(data)


client1 = SocketClass()
client2 = SocketClass()

client2.threaded_client(client1.createSocket(), client2.createSocket())
