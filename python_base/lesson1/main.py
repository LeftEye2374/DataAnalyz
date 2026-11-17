import asyncio
import time


async def fetch_data(seconds):
    print("Начинаем запрос")
    await asyncio.sleep(seconds)
    print("Запрос прошел успешно")
    return 42

async def main():
    print(await fetch_data(2))
    print(await fetch_data(4))

asyncio.run(main())
