import socket
import subprocess
import cv2

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(("0.0.0.0", 8000))
server_socket.listen(0)

cv2.VideoCapture()

connection = server_socket.accept()[0].makefile("rb")

try:

    while True:

        data = connection.read(1024)
        cv2.imdecode()


finally:
    connection.close()
    server_socket.close()
