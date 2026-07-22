from .local_cpu import LocalCPUBackend


class LocalGPUBackend(LocalCPUBackend):
    """Same model as local_cpu on CUDA. Requires an NVIDIA GPU on the worker
    host, the nvidia container toolkit, and a torch build with CUDA (swap the
    CPU-wheel line in worker/Dockerfile)."""

    name = "local_gpu"
    device = "cuda"
