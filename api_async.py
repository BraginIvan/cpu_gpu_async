import random
from concurrent.futures.process import ProcessPoolExecutor
from fastapi import FastAPI, File
from inference import Cpu, Gpu
from multiprocessing import Queue, Manager
import asyncio
from pydantic import BaseModel
from typing import List

cpu = Cpu()

def cpu_processing(batch, gpu_queue:Queue, req_id):
    result = cpu.pre_process(batch)
    gpu_queue.put((req_id, result))

app = FastAPI()
manager = Manager()
gpu_queue = manager.Queue()
prediction_queue = manager.Queue()

def input_queue_listener(input_queue: Queue, output_queue: Queue):
    gpu = Gpu()
    while True:
        key, batch = input_queue.get()
        preds = gpu.process(batch)
        output_queue.put((key, preds))

class Item(BaseModel):
    data: List
    id: int

results = {}

async def read_predict(id):
    for _ in range(100):
        await asyncio.sleep(1/10)
        if id in results:
            return results.pop(id)
        try:
            key, value = prediction_queue.get(block=False)
            if key == id:
                return value
            else:
                results[key]=value
        except:
            pass
    return -1

@app.post("/predictions/resnet-18")
async def predict(data: list[bytes] = File(...)):
    id = random.randint(0, 1000000)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(app.state.executor, cpu_processing, data, gpu_queue, id)
    res = await read_predict(id)
    print(res)
    return cpu.post_process(res)


@app.on_event("startup")
async def startup_event():
    app.state.executor = ProcessPoolExecutor(max_workers=3)
    loop = asyncio.new_event_loop()
    loop.run_in_executor(ProcessPoolExecutor(max_workers=1), input_queue_listener, gpu_queue, prediction_queue)


@app.on_event("shutdown")
async def on_shutdown():
    app.state.executor.shutdown()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
    pass

