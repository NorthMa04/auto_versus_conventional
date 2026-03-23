# gpu_bench.py
import time, torch, numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device, torch.cuda.get_device_name(0) if device=='cuda' else '')
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
# 预热
for _ in range(5):
    _ = a @ b
if device=='cuda': torch.cuda.synchronize()
times=[]
for _ in range(5):
    t0=time.perf_counter()
    _ = a @ b
    if device=='cuda': torch.cuda.synchronize()
    times.append(time.perf_counter()-t0)
print('times', times)
print('mean', np.mean(times))
