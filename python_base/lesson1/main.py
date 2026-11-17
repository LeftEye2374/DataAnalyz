import asyncio
import time


async def fetch_data(seconds):
    print("Начинаем запрос")
    await asyncio.sleep(seconds)
    print("Запрос прошел успешно")
    return 42

async def main():
    start = time.perf_counter()
   
    await asyncio.gather(fetch_data(2), fetch_data(4), fetch_data(5))

    print(time.perf_counter() - start)

asyncio.run(main())
