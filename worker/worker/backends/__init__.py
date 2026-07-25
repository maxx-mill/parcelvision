from .base import Backend


def get_backend(name: str) -> Backend:
    """Resolve INFERENCE_BACKEND to an implementation. ML-heavy backends import
    torch/geoai lazily inside extract(), so resolution itself is always cheap."""
    if name == "local_cpu":
        from .local_cpu import LocalCPUBackend

        return LocalCPUBackend()
    if name == "local_gpu":
        from .local_gpu import LocalGPUBackend

        return LocalGPUBackend()
    if name == "rfdetr":
        from .rfdetr import RFDetrBackend

        return RFDetrBackend()
    if name == "endpoint":
        from .endpoint import EndpointBackend

        return EndpointBackend()
    if name == "fake":
        from .fake import FakeBackend

        return FakeBackend()
    raise ValueError(f"unknown INFERENCE_BACKEND {name!r}")
