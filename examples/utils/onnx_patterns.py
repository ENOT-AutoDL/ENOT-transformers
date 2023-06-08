from typing import List
from typing import Set

import onnx


def find_inputs(model_proto: onnx.ModelProto, dst_node: onnx.NodeProto) -> List[onnx.NodeProto]:
    mapping = {output: node for node in model_proto.graph.node for output in node.output}
    return [mapping[node_input] for node_input in dst_node.input if node_input in mapping]


def find_outputs(model_proto: onnx.ModelProto, src_node: onnx.NodeProto) -> List[onnx.NodeProto]:
    mapping = {}
    for node in model_proto.graph.node:
        for node_input in node.input:
            mapping.setdefault(node_input, []).append(node)

    result = []
    for node_output in src_node.output:
        if node_output in mapping:
            result += mapping[node_output]

    return result


def find_all_gelu_elements(onnx_proto) -> Set[str]:
    result = []
    for node in onnx_proto.graph.node:
        if node.op_type == 'Tanh':
            # Tanh node
            result.append(node.name)
            # Mul node
            (node_0,) = find_inputs(onnx_proto, node)
            result.append(node_0.name)
            # Add node
            (node_0,) = find_inputs(onnx_proto, node_0)
            result.append(node_0.name)
            # Mul node
            node_0, node_1 = find_inputs(onnx_proto, node_0)
            node_0 = node_0 if node_0.op_type == 'Mul' else node_1
            result.append(node_0.name)
            # Pow node
            (node_0,) = find_inputs(onnx_proto, node_0)
            result.append(node_0.name)
            # Add node
            (node_0,) = find_outputs(onnx_proto, node)
            result.append(node_0.name)
            # Mul node
            (node_0,) = find_outputs(onnx_proto, node_0)
            result.append(node_0.name)
            # Add & Mul node
            node_0, node_1 = find_inputs(onnx_proto, node_0)
            result += [node_0.name, node_1.name]

    return set(result)


def find_all_layernorm_elements(onnx_proto) -> Set[str]:
    result = []
    for node in onnx_proto.graph.node:
        if node.op_type == 'ReduceMean':
            (node_0,) = find_outputs(onnx_proto, node)
            if node_0.op_type != 'Sub':
                continue

            # first ReduceMean node and Sub node
            result.append(node.name)
            result.append(node_0.name)
            # Div and Pow node
            node_0, node_1 = find_outputs(onnx_proto, node_0)
            result.append(node_0.name)
            result.append(node_1.name)
            node_0, node_1 = (node_0, node_1) if node_0.op_type == 'Pow' else (node_1, node_0)
            # second ReduceMean node
            (node_0,) = find_outputs(onnx_proto, node_0)
            result.append(node_0.name)
            # Add node
            (node_0,) = find_outputs(onnx_proto, node_0)
            result.append(node_0.name)
            # Sqrt node
            (node_0,) = find_outputs(onnx_proto, node_0)
            result.append(node_0.name)
            # Mul node
            (node_1,) = find_outputs(onnx_proto, node_1)
            result.append(node_1.name)
            # Add node
            (node_1,) = find_outputs(onnx_proto, node_1)
            result.append(node_1.name)

    return set(result)
