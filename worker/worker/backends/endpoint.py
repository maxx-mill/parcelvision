from pathlib import Path

import geopandas as gpd


class EndpointBackend:
    """Hosted-inference stub: POST imagery to a GPU endpoint (Hugging Face
    Inference Endpoints, Modal, a box with FastAPI in front of the same model)
    and parse returned GeoJSON. The config toggle exists so swapping compute is
    an env change, not a rewrite; wiring it up is deliberately out of MVP scope."""

    name = "endpoint"
    needs_imagery = True

    def extract(self, raster_paths: list[Path], bbox: list[float]) -> gpd.GeoDataFrame:
        raise NotImplementedError(
            "INFERENCE_BACKEND=endpoint is a documented stub — set INFERENCE_ENDPOINT_URL "
            "and implement the client here when a hosted endpoint exists. "
            "Use local_cpu (default) or local_gpu meanwhile."
        )
