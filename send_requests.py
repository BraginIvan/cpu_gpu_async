from pathlib import Path
import aiohttp
import asyncio
import time
import more_itertools
from utils.client_args import get_args

def gen_port(ports):
    while True:
        yield from ports

async def make_request(session, batch, port):
    async with session.post(f'http://127.0.0.1:{port}/predictions/resnet',
                            data=batch) as resp:
        return await resp.text()

async def main(paths):
    async with aiohttp.ClientSession() as session:
            tasks = []
            for batch in images_batches:
                tasks.append(asyncio.ensure_future(make_request(session, batch, next(port))))
            predicts = await asyncio.gather(*tasks)
            for predict, path in zip(predicts, paths):
                print(predict)
                break





if __name__ == '__main__':
    args = get_args()
    images_path = Path(args.imgs_path)
    images_n = args.images_n
    BATCH_SIZE = args.batch_size
    ports = list(range(8080, 8080+args.ports))
    port = gen_port(ports)
    paths = list(images_path.glob("*"))
    if len(paths) < images_n:
        paths = paths * (images_n // len(paths) + 2)
    paths = paths[:images_n]
    times = []
    for i in range(20):
        images = [("data", open(str(path), "rb")) for path in paths]
        images_batches = list(more_itertools.grouper(images, BATCH_SIZE))
        start_time = time.time()
        asyncio.run(main(images_batches))
        spent = time.time() - start_time
        times.append(spent)
    times = sorted(times)
    print(f"mean = {sum(times) / 20}, confidence interval [{times[1]}, {times[-2]}]")



