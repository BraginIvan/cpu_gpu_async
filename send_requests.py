from pathlib import Path
import aiohttp
import asyncio
import time
import more_itertools

# folder with images
images_path = Path("/home/ivan/tmp")
images_n = 1608
BATCH_SIZE = 24

def gen_port():
    while True:
        yield from [8080, 8081, 8082]
        # yield from [8080]

port = gen_port()

async def make_request(session, batch, port = 8080):
    async with session.post(f'http://127.0.0.1:{port}/predictions/resnet-18',
                            data=batch) as resp:
        return await resp.text()

async def main():
    async with aiohttp.ClientSession() as session:
        paths = list(images_path.glob("*"))
        if len(paths) < images_n:
            paths = paths * (images_n // len(paths)+2 )
        paths=paths[:images_n]
        times = []
        for i in range(20):
            images = [("data", open(str(path), "rb")) for path in paths]
            images_batches=list(more_itertools.grouper(images, BATCH_SIZE))

            start_time = time.time()
            tasks = []
            for batch in images_batches:
                tasks.append(asyncio.ensure_future(make_request(session, batch, next(port))))

            predicts = await asyncio.gather(*tasks)
            for predict, path in zip(predicts, paths):
                print(predict, path)
                break

            spent = time.time() - start_time
            times.append(spent)
            print(f"--- {spent} seconds ---")
        times = sorted(times)
        print(f"min = {times[0]}, max = {times[-1]}, mean = {sum(times)/20}, confidence interval [{times[1]}, {times[-2]}]")

asyncio.run(main())

