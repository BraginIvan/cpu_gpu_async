from fastapi import FastAPI, File
import uvicorn
from inference.inference import Cpu, Gpu
from multiprocessing import Process, Manager, Queue
import random

cpu = Cpu()
gpu = Gpu()
manager = Manager()
gpu_queue = manager.Queue()
ports = [8080, 8081, 8082]
out_queues = {port: manager.Queue() for port in ports}

def input_queue_listener(input_queue: Queue, output_queues: dict[Queue]):
    gpu = Gpu()
    while True:
        key, port, batch = input_queue.get()
        preds = gpu.process(batch)
        output_queues[port].put((key, preds))

def start_rest(port: int):
    app = get_app(gpu_queue, out_queues[port], port)
    uvicorn.run(app, host="127.0.0.1", port=port)

def get_app(gpu_queue: Queue, out_queue: Queue, port: int):

    app = FastAPI()
    @app.post("/predictions/resnet-18")
    async def predict(data: list[bytes] = File(...)):
        id = random.randint(0, 1000000)
        res = cpu.pre_process(data)
        gpu_queue.put((id, port, res))
        key, value = out_queue.get(block=False)
        return cpu.post_process(value)
    return app


if __name__ == "__main__":
    for port in ports:
        p = Process(target=start_rest, args=(port,))
        p.start()
    input_queue_listener(gpu_queue, out_queues)



