from fastapi import FastAPI, File
import uvicorn
from inference import Cpu, Gpu
from multiprocessing import Process, Manager, Queue
import asyncio

cpu = Cpu()
gpu = Gpu()
manager = Manager()
gpu_queue = manager.Queue()
ports = [8080, 8081, 8082, 8083, 8084]
out_queues = {port: manager.Queue() for port in ports}

def input_queue_listener(input_queue: Queue, output_queues: dict[Queue]):
    gpu = Gpu()
    while True:
        key, port, batch = input_queue.get()
        print("gpu", key)
        preds = gpu.process(batch)
        print("queue", key)
        output_queues[port].put((key, preds))
        print("finish", key)

def start_rest(port: int):
    app = get_app(gpu_queue, out_queues[port], port)
    uvicorn.run(app, host="127.0.0.1", port=port)

def get_app(gpu_queue: Queue, out_queue: Queue, port: int):
    results = {}

    async def read_predict(id):

        for approach in range(100):
            await asyncio.sleep(1 / 10)
            if id in results:
                return results.pop(id)
            print(id, approach, results.keys())
            try:
                key, value = out_queue.get(block=False)
                if key == id:
                    print("skip", id)
                    return value
                else:
                    results[key] = value
                    print("load", approach, key, results.keys())
            except:
                pass
        print("lost", id, results.keys())
        return -1

    app = FastAPI()
    @app.post("/predictions/resnet-18")
    async def predict(data: list[bytes] = File(...)):
        id = hash(data[0])
        res = cpu.pre_process(data)
        gpu_queue.put((id, port, res))
        res = await read_predict(id)
        return cpu.post_process(res)
    return app

if __name__ == "__main__":
    for port in ports:
        p = Process(target=start_rest, args=(port,))
        p.start()
    input_queue_listener(gpu_queue, out_queues)



