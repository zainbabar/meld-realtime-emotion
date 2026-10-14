# smoke test: can this torch build actually run cuda kernels on this gpu?
# checks cuBLAS (matmul), pytorch's own compiled kernels (softmax, elementwise)
# and cuDNN (conv1d), each against a cpu reference
#
# usage:
#   uv run python -I scripts/check_gpu.py
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


# relative error of a gpu result against a cpu reference, both in fp64
def rel_err(out, ref):
    diff = out.double().cpu() - ref.double()
    return (diff.norm() / ref.double().norm()).item()


torch.manual_seed(0)
n = 4096
a, b = torch.randn(n, n), torch.randn(n, n)
ref = a.double() @ b.double()

print("\nmatmul (cuBLAS), 4096x4096, vs fp64 CPU reference:")
for dtype in (torch.float32, torch.bfloat16):
    ga, gb = a.to(dev, dtype), b.to(dev, dtype)
    out = ga @ gb  # warm up, the first call pays one-time setup costs
    # cuda calls are async, so sync before reading the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        out = ga @ gb
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    tflops = 2 * n**3 / dt / 1e12  # n^3 multiply-adds = 2n^3 flops
    print(f"  {str(dtype):15s} rel err {rel_err(out, ref):.1e}   {tflops:6.1f} TFLOP/s")

print("\nnative PyTorch kernels and cuDNN, vs CPU float32:")
x = torch.randn(64, 1000)
print(f"  softmax          rel err {rel_err(torch.softmax(x.to(dev), -1), torch.softmax(x, -1)):.1e}")
gelu = torch.nn.functional.gelu
print(f"  gelu(x)*2+1      rel err {rel_err(gelu(x.to(dev)) * 2 + 1, gelu(x) * 2 + 1):.1e}")
# same shape as a wav2vec-style first conv layer: 25 ms windows, 10 ms hop at 16 kHz
conv = torch.nn.Conv1d(1, 64, kernel_size=400, stride=160)
wav = torch.randn(8, 1, 16000)
print(f"  conv1d (cuDNN)   rel err {rel_err(conv.to(dev)(wav.to(dev)), conv.cpu()(wav)):.1e}")
print("\nGPU OK")
