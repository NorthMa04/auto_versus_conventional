import time
import multiprocessing
import numpy as np

# 定义一个计算密集型任务：矩阵乘法
def matmul_task(size: int, repeat: int):
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    for _ in range(repeat):
        np.dot(A, B)  # 矩阵乘法

if __name__ == "__main__":
    size = 800   # 矩阵大小，可以调大
    repeat = 200  # 重复次数，可以调大

    # 单核测试
    start = time.time()
    matmul_task(size, repeat)
    end = time.time()
    print(f"单核耗时: {end - start:.2f} 秒")

    # 多核测试
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU核心数: {cpu_count}")

    start = time.time()
    with multiprocessing.Pool(processes=cpu_count) as pool:
        pool.starmap(matmul_task, [(size, repeat // cpu_count)] * cpu_count)
    end = time.time()
    print(f"多核耗时: {end - start:.2f} 秒")
