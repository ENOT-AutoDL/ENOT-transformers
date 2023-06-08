# Transformer inference on TensorRT with INT-8 precision

Repository contains inference example and accuracy validation of quantized transformer TensorRT models.
All onnx models are published on Hugging Face :hugs::
* [GPT2-XL](https://huggingface.co/ENOT-AutoDL/gpt2-tensorrt)
* [GPT-J 6B](https://huggingface.co/ENOT-AutoDL/gpt-j-6B-tensorrt-int8)

Our example notebooks automatically download the appropriate onnx and build engine.

## Metrics:

### GPT2-XL

|   |TensorRT INT8+FP32|torch FP16|
|---|:---:|:---:|
| **Lambada Acc** |72.11%|71.43%|
| **Model size (GB)** |2.0|3.2|

### GPT-J 6B

|   |TensorRT INT8+FP32|torch FP16|torch FP32|
|---|:---:|:---:|:---:|
| **Lambada Acc** |78.46%|79.53%|-|
| **Model size (GB)**  |8.5|12.1|24.2|

### Test environment

* GPU RTX 4090
* CPU 11th Gen Intel(R) Core(TM) i7-11700K
* TensorRT 8.5.3.1
* pytorch 1.13.1+cu116

## Latency:

### GPT2-XL

|Input sequance length|Number of generated tokens|TensorRT INT8+FP32 ms|torch FP16 ms|Acceleration|
|:---:|:---:|:---:|:---:|:---:|
|64|64|462|1190|2.58|
|64|128|920|2360|2.54|
|64|256|1890|4710|2.54|

### GPT-J 6B

|Input sequance length|Number of generated tokens|TensorRT INT8+FP32 ms|torch FP16 ms|Acceleration|
|:---:|:---:|:---:|:---:|:---:|
|64|64|1040|1610|1.55|
|64|128|2089|3224|1.54|
|64|256|4236|6479|1.53|

### Test environment

* GPU RTX 4090
* CPU 11th Gen Intel(R) Core(TM) i7-11700K
* TensorRT 8.5.3.1
* pytorch 1.13.1+cu116
