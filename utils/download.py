from pathlib import Path
from typing import Union

from huggingface_hub import hf_hub_download

from utils.trt_builder import DefaultTransformerEngineBuilder


def download_model_onnx(model_name: str, repo_id: str, cache_dir: Union[str, Path]) -> Path:
    hf_hub_download(
        repo_id=repo_id,
        filename=f'{model_name}.data',
        local_dir=cache_dir,
    )
    path_to_onnx = hf_hub_download(
        repo_id=repo_id,
        filename=f'{model_name}.onnx',
        local_dir=cache_dir,
    )

    return Path(path_to_onnx)
