from fastapi import FastAPI, File
import uvicorn
from utils.inference_socket import Cpu, Gpu
from multiprocessing import Process, Manager
import socket
import selectors
import numpy as np
from utils.serving_args import get_args

selector = selectors.DefaultSelector()

cpu = Cpu()
manager = Manager()

def recvall(sock):
    data = bytearray()
    while True:
        packet = sock.recv(65536)
        if not packet:
            return data
        data.extend(packet)

localhost = "127.0.0.1"

def gpu_listener():
    gpu = Gpu(args.model)

    def accept_connection(server_socket):
        client_socket, addr = server_socket.accept()
        selector.register(fileobj=client_socket, events=selectors.EVENT_READ, data=gpu_process)

    def gpu_process(client_socket):
        batch = recvall(client_socket)
        preds = gpu.process(batch)
        client_socket.send(preds)
        client_socket.close()
        selector.unregister(client_socket)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((localhost, 5000))
    server_socket.listen()
    selector.register(fileobj=server_socket, events=selectors.EVENT_READ, data=accept_connection)
    while True:
        events = selector.select()
        for key, _ in events:
            callback = key.data
            callback(key.fileobj)


def start_rest(port: int):
    app = get_app()
    uvicorn.run(app, host=localhost, port=port)


def get_app():
    app = FastAPI()

    @app.post("/predictions/resnet")
    async def predict(data: list[bytes] = File(...)):
        res = cpu.pre_process(data)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((localhost, 5000))
        s.sendall(res.tobytes())
        s.shutdown(socket.SHUT_WR)
        data = recvall(s)
        data = np.frombuffer(data, dtype="float32")
        data = data.reshape(-1, 1000)
        s.close()
        return cpu.post_process(data)

    return app


if __name__ == "__main__":
    args = get_args()
    ports = list(range(8080, 8080 + args.ports))
    for port in ports:
        p = Process(target=start_rest, args=(port,))
        p.start()
    gpu_listener()
