import logging
from pathlib import Path
from typing import List
from typing import Union

import onnx
import tensorrt as trt

from utils.onnx_patterns import find_all_gelu_elements
from utils.onnx_patterns import find_all_layernorm_elements
from utils.trt_logger import TRT_LOGGER

_LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_WORKSPACE_SIZE = 15 * 1024 * 1024 * 1024


class BaseEngineBuilder:
    def __init__(self, use_fp16: bool, use_int8: bool, max_workspace_size: int = DEFAULT_MAX_WORKSPACE_SIZE):
        self._use_fp16 = use_fp16
        self._use_int8 = use_int8
        self._max_workspace_size = max_workspace_size

    def _build_network(self, builder: trt.Builder, path_to_onnx: str) -> trt.INetworkDefinition:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
        onnx_parser.parse_from_file(path_to_onnx)

        if self._use_fp16:
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                if 'history' in input_tensor.name:
                    _LOGGER.info(f'FIX INPUT TYPE TO FP16 -> "{input_tensor.name}"')
                    input_tensor.dtype = trt.DataType.HALF

            for i in range(network.num_outputs):
                output_tensor = network.get_output(i)
                if 'history' in output_tensor.name:
                    _LOGGER.info(f'FIX OUTPUT TYPE TO FP16 -> "{output_tensor.name}"')
                    output_tensor.dtype = trt.DataType.HALF

        return network

    def _build_config(self, builder: trt.Builder) -> trt.IBuilderConfig:
        config = builder.create_builder_config()
        config.max_workspace_size = self._max_workspace_size

        config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)
        if self._use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if self._use_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        return config

    def build(self, path_to_onnx: Union[str, Path], engine_cache_path: Union[str, Path]) -> None:
        builder = trt.Builder(TRT_LOGGER)
        network = self._build_network(builder, str(path_to_onnx))
        config = self._build_config(builder)

        engine = builder.build_engine(network, config)
        with open(str(engine_cache_path), 'wb') as engine_cache:
            engine_cache.write(engine.serialize())


class TransformerEngineBuilder(BaseEngineBuilder):
    def __init__(
        self,
        min_batch_size: int,
        opt_batch_size: int,
        max_batch_size: int,
        min_seq_len: int,
        opt_seq_len: int,
        max_seq_len: int,
        min_history_len: int,
        opt_history_len: int,
        max_history_len: int,
        use_fp16: bool = False,
        use_int8: bool = False,
        force_ln_fp32: bool = True,
        force_gelu_fp32: bool = False,
        max_workspace_size: int = DEFAULT_MAX_WORKSPACE_SIZE,
    ):
        super().__init__(
            use_fp16=use_fp16,
            use_int8=use_int8,
            max_workspace_size=max_workspace_size,
        )

        self._force_ln_fp32 = force_ln_fp32
        self._force_gelu_fp32 = force_gelu_fp32

        self._min_batch_size = min_batch_size
        self._opt_batch_size = opt_batch_size
        self._max_batch_size = max_batch_size

        self._min_seq_len = min_seq_len
        self._opt_seq_len = opt_seq_len
        self._max_seq_len = max_seq_len

        self._min_history_len = min_history_len
        self._opt_history_len = opt_history_len
        self._max_history_len = max_history_len

        self._history_inputs: List[str] = []
        self._num_heads = None
        self._hidden_size = None

    def _build_network(self, builder: trt.Builder, path_to_onnx: str) -> trt.INetworkDefinition:
        network = super()._build_network(builder, path_to_onnx)
        self._history_inputs = []
        for i in range(network.num_inputs):
            name = network.get_input(i).name
            if 'history' in name:
                self._history_inputs.append(name)
                if self._num_heads is None:
                    _, self._num_heads, _, self._hidden_size = network.get_input(i).shape

        if self._force_gelu_fp32 or self._force_ln_fp32:
            onnx_model = onnx.load(path_to_onnx, load_external_data=False)
            fp32_elements = set()
            if self._force_gelu_fp32:
                fp32_elements |= find_all_gelu_elements(onnx_model)
            if self._force_ln_fp32:
                fp32_elements |= find_all_layernorm_elements(onnx_model)

            for i in range(len(network)):
                layer = network[i]
                if layer.name in fp32_elements:
                    layer.precision = trt.DataType.FLOAT
                    layer.set_output_type(0, trt.DataType.FLOAT)

        return network

    def _build_config(self, builder: trt.Builder) -> trt.IBuilderConfig:
        config = super()._build_config(builder)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            'input_ids',
            min=(self._min_batch_size, self._min_seq_len),
            opt=(self._opt_batch_size, self._opt_seq_len),
            max=(self._max_batch_size, self._max_seq_len),
        )
        for name in self._history_inputs:
            profile.set_shape(
                name,
                min=(self._min_batch_size, self._num_heads, self._min_history_len, self._hidden_size),
                opt=(self._opt_batch_size, self._num_heads, self._opt_history_len, self._hidden_size),
                max=(self._max_batch_size, self._num_heads, self._max_history_len, self._hidden_size),
            )
        config.add_optimization_profile(profile)

        return config


class DefaultTransformerEngineBuilder(TransformerEngineBuilder):
    def __init__(
        self,
        max_batch_size: int = 1,
        max_seq_len: int = 256,
        max_history_len: int = 512,
        use_fp16: bool = False,
        use_int8: bool = False,
        max_workspace_size: int = DEFAULT_MAX_WORKSPACE_SIZE,
    ):
        super().__init__(
            min_batch_size=1,
            opt_batch_size=max_batch_size,
            max_batch_size=max_batch_size,
            min_seq_len=1,
            opt_seq_len=1,
            max_seq_len=max_seq_len,
            min_history_len=0,
            opt_history_len=max_history_len,
            max_history_len=max_history_len,
            use_fp16=use_fp16,
            use_int8=use_int8,
            force_ln_fp32=use_fp16,
            force_gelu_fp32=False,
            max_workspace_size=max_workspace_size,
        )
