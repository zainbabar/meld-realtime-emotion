"""Smoke test: can this torch build actually run CUDA kernels on this GPU?

Checks cuBLAS (matmul), PyTorch's own compiled kernels (softmax, elementwise),
and cuDNN (conv1d), each compared against a CPU reference.
"""
import time

import torch

print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(1)

dev = torch.device("cuda")
major, minor = torch.cuda.get_device_capability(0)
print(f"device: {torch.cuda.get_device_name(0)} | compute capability sm_{major}{minor}")
print(f"archs compiled into this build: {torch.cuda.get_arch_list()}")


def rel_err(out, ref):
    return ((out.double().cpu() - ref.double()).norm() / ref.double().norm()).item()


torch.manual_seed(0)
n = 4096
a, b = torch.randn(n, n), torch.randn(n, n)
ref = a.double() @ b.double()

print("\nmatmul (cuBLAS), 4096x4096, vs fp64 CPU reference:")
for dtype in (torch.float32, torch.bfloat16):
    ga, gb = a.to(dev, dtype), b.to(dev, dtype)
    out = ga @ gb
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        out = ga @ gb
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    print(f"  {str(dtype):15s} rel err {rel_err(out, ref):.1e}   {2 * n**3 / dt / 1e12:6.1f} TFLOP/s")

print("\nnative PyTorch kernels and cuDNN, vs CPU float32:")
x = torch.randn(64, 1000)
print(f"  softmax          rel err {rel_err(torch.softmax(x.to(dev), -1), torch.softmax(x, -1)):.1e}")
print(f"  gelu(x)*2+1      rel err {rel_err(torch.nn.functional.gelu(x.to(dev)) * 2 + 1, torch.nn.functional.gelu(x) * 2 + 1):.1e}")
conv = torch.nn.Conv1d(1, 64, kernel_size=400, stride=160)
wav = torch.randn(8, 1, 16000)
print(f"  conv1d (cuDNN)   rel err {rel_err(conv.to(dev)(wav.to(dev)), conv.cpu()(wav)):.1e}")
print("\nGPU OK")
