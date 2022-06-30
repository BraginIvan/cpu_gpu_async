import random
from concurrent.futures.process import ProcessPoolExecutor
from fastapi import FastAPI, File
from utils.inference import Cpu, Gpu
from multiprocessing import Queue, Manager
import asyncio
from utils.serving_args import get_args

app = FastAPI()

def cpu_processing(batch, gpu_queue:Queue, req_id: int):
    result = cpu.pre_process(batch)
    gpu_queue.put((req_id, result))

def input_queue_listener(input_queue: Queue, output_queue: Queue, model: str):
    gpu = Gpu(model)
    while True:
        key, batch = input_queue.get()
        preds = gpu.process(batch)
        output_queue.put((key, preds))

results = {}
async def read_predict(id):
    for approach in range(100):
        await asyncio.sleep(1 / 10)
        if id in results:
            return results.pop(id)
        try:
            key, value = prediction_queue.get(block=False)
            if key == id:
                return value
            else:
                results[key] = value
        except:
            pass

@app.post("/predictions/resnet")
async def predict(data: list[bytes] = File(...)):
    id = random.randint(0, 1000000)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(app.state.executor, cpu_processing, data, gpu_queue, id)
    res = await read_predict(id)
    return cpu.post_process(res)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.new_event_loop()
    gpu_process = ProcessPoolExecutor(max_workers=1)
    loop.run_in_executor(gpu_process, input_queue_listener, gpu_queue, prediction_queue, args.model)
    app.state.executor = ProcessPoolExecutor(max_workers=3)

@app.on_event("shutdown")
async def on_shutdown():
    app.state.executor.shutdown()

if __name__ == "__main__":
    args = get_args()
    manager = Manager()
    gpu_queue = manager.Queue()
    prediction_queue = manager.Queue()
    cpu = Cpu()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)


