import numpy as np
import pytest
import torch

from fod_pipeline.classifier.model import MLP2Classifier
from fod_pipeline.mobile.convert_utils import export_onnx
from fod_pipeline.mobile.wrappers import ClassifierWithSoftmax


def test_export_onnx_produces_a_loadable_graph_with_expected_io(tmp_path):
    onnx = pytest.importorskip("onnx")

    model = ClassifierWithSoftmax(MLP2Classifier(input_dim=512, num_classes=4))
    dummy_input = torch.randn(1, 512)
    onnx_path = tmp_path / "classifier.onnx"

    export_onnx(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["embedding"],
        output_names=["class_probabilities"],
    )

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)

    input_names = [i.name for i in graph.graph.input]
    output_names = [o.name for o in graph.graph.output]
    assert input_names == ["embedding"]
    assert output_names == ["class_probabilities"]


def test_exported_onnx_graph_matches_pytorch_output(tmp_path):
    ort = pytest.importorskip("onnxruntime")

    torch_model = ClassifierWithSoftmax(MLP2Classifier(input_dim=512, num_classes=4))
    torch_model.eval()
    dummy_input = torch.randn(1, 512)
    onnx_path = tmp_path / "classifier.onnx"

    export_onnx(
        torch_model,
        dummy_input,
        str(onnx_path),
        input_names=["embedding"],
        output_names=["class_probabilities"],
    )

    x = torch.randn(1, 512)
    with torch.no_grad():
        torch_out = torch_model(x).numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(["class_probabilities"], {"embedding": x.numpy()})[0]

    # ONNX Runtime executes the same erf-based GELU as PyTorch (no
    # TFLite-style approximation needed), so this should be near bit-exact.
    assert np.abs(torch_out - onnx_out).max() < 1e-5
