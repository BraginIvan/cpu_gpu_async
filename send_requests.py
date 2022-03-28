import random
from pathlib import Path
import aiohttp
import asyncio
import time
import more_itertools

# folder with images
images_path = Path("/home/ivan/projects/ICU/auto_train/dog/")
images = 1600
# for torchserve use BATCH_SIZE=1
BATCH_SIZE = 4

def gen_port():
    while True:
        yield from [8080, 8081, 8082, 8083, 8084]
        # yield from [8080]

port = gen_port()

async def make_request(session, batch, port = 8080):
    async with session.post(f'http://127.0.0.1:{port}/predictions/resnet-18',
                            data=[("data", open(str(path), "rb")) for path in batch]) as resp:
        return await resp.text()

async def main():
    async with aiohttp.ClientSession() as session:
        paths = list(images_path.glob("*"))[:images]
        paths_batches = more_itertools.grouper(paths, BATCH_SIZE)
        tasks = []
        for batch in paths_batches:
            tasks.append(asyncio.ensure_future(make_request(session, batch, next(port))))
        predicts = await asyncio.gather(*tasks)
        for predict in predicts:
            print(predict)

start_time = time.time()
asyncio.run(main())
print("--- %s seconds ---" % (time.time() - start_time))


# --- 8.342940330505371 seconds --- async
# --- 20.39217185974121 seconds --- sync
# --- 5.913234233856201 seconds --- torchserve