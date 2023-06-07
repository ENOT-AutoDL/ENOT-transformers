from pathlib import Path
from typing import Union

import torch

from utils.trt_model import TrtModel


class TrtSeq2SeqModel:
    def __init__(self, path_to_engine: Union[Path, str], external_history_concat: bool = True):
        self._model = TrtModel(str(path_to_engine))
        self._external_history_concat = external_history_concat

        self._num_heads = None
        self._hidden_size = None

    def generate(self, input_ids: torch.Tensor, generate_len: int, return_logit: bool = False) -> torch.Tensor:
        input_ids = input_ids.contiguous()

        input_tensors = {'input_ids': input_ids}
        for name in self._model.inputs:
            if name.startswith('history'):
                if self._num_heads is None:
                    _, self._num_heads, _, self._hidden_size = self._model.binding_shape(name)

                input_tensors[name] = torch.empty(
                    size=(input_ids.size(0), self._num_heads, 0, self._hidden_size),
                    dtype=self._model.binding_dtype(name),
                    device='cuda',
                )

        result = []
        output_tensors = None
        for _ in range(generate_len):
            output_tensors = self._model.run(input_tensors=input_tensors, output_tensors_cache=output_tensors)

            logits = output_tensors['logits']
            next_id = logits[:, -1, :].argmax(dim=-1, keepdims=True).to(torch.int32)
            result.append(logits.clone() if return_logit else next_id)

            input_tensors['input_ids'] = next_id
            for name, new_value in output_tensors.items():
                if name.startswith('out_history_'):
                    name = name[4:]
                    if self._external_history_concat:
                        input_tensors[name] = torch.cat((input_tensors[name], new_value), dim=-2)
                    else:
                        input_tensors[name] = new_value

        dim = -2 if return_logit else -1
        result = torch.cat(result, dim=dim)

        return result
