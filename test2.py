import asyncio

async def get_data(name):
    print(f"{name}: 開始")
    await asyncio.sleep(1)  # 模擬等待網路回應
    print(f"{name}: 完成")
    return f"{name} 的資料"

async def main():
    results = await asyncio.gather(
        get_data("任務 A"),
        get_data("任務 B")
    )
    print(results)

asyncio.run(main())