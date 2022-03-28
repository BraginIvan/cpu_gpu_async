from fastapi import FastAPI, File
import uvicorn
from inference import Cpu, Gpu

app = FastAPI()
cpu = Cpu()
gpu = Gpu()

@app.post("/predictions/resnet-18")
async def predict(data: list[bytes] = File(...)):
    res = gpu.process(cpu.pre_process(data))
    return cpu.post_process(res)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
