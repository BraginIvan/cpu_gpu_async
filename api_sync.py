from fastapi import FastAPI, File
import uvicorn
from utils.inference import Cpu, Gpu
from utils.serving_args import get_args

app = FastAPI()


@app.post("/predictions/resnet")
async def predict(data: list[bytes] = File(...)):
    preprocessed = cpu.pre_process(data)
    predicted = gpu.process(preprocessed)
    postprocessed = cpu.post_process(predicted)
    return postprocessed


if __name__ == "__main__":
    args = get_args()
    cpu = Cpu()
    gpu = Gpu(args.model)
    uvicorn.run(app, host="127.0.0.1", port=8080)
