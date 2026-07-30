"""
Shared PyTorch -> ONNX / CoreML conversion helpers.

Every mobile export in this package follows one of two paths out of
PyTorch:

    PyTorch model --(torch.onnx.export)-------------------------> ONNX      [Android, via ONNX Runtime Mobile]
    PyTorch model --(torch.jit.trace)---------------(coremltools)--> CoreML  [iOS]
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import torch


def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    input_names: list,
    output_names: list,
    opset: int = 17,
) -> str:
    """Trace `model` and write it out as a static-shape ONNX graph.

    A static input shape is used deliberately: mobile
    inference always runs on a single image (batch size 1), and fixed
    shapes convert far more reliably than dynamic ones.
    """
    model = model.eval()
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


def torch_to_coreml(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_name: str,
    output_name: str,
    minimum_deployment_target: str = "iOS16",
) -> str:
    """Trace `model` and convert it directly to Core ML.
    """
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError(
            "coremltools is required for Core ML export. Install it with: "
            "pip install -e '.[mobile]'"
        ) from exc

    model = model.eval()
    traced = torch.jit.trace(model, dummy_input)
    inputs = [ct.TensorType(name=input_name, shape=tuple(dummy_input.shape), dtype=np.float32)]
    outputs = [ct.TensorType(name=output_name, dtype=np.float32)]

    try:
        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=outputs,
            convert_to="mlprogram",
            minimum_deployment_target=getattr(ct.target, minimum_deployment_target),
        )
    except RuntimeError as exc:
        if "BlobWriter" not in str(exc):
            raise
        warnings.warn(
            "coremltools could not write an .mlpackage (mlprogram) on this "
            "platform - its native weight-serialization extension is only "
            "shipped prebuilt for macOS/Linux. Falling back to the legacy "
            ".mlmodel (neuralnetwork) format, which Xcode/Core ML still "
            "fully support, just without newer mlprogram-only "
            "quantization options. Run the export on macOS or Linux to "
            "get a .mlpackage instead.",
            stacklevel=2,
        )
        output_path = os.path.splitext(output_path)[0] + ".mlmodel"
        mlmodel = ct.convert(traced, inputs=inputs, outputs=outputs, convert_to="neuralnetwork")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mlmodel.save(output_path)
    return output_path
