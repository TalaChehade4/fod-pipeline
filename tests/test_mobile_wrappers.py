import torch

from fod_pipeline.classifier.model import MLP2Classifier
from fod_pipeline.mobile.wrappers import ClassifierWithSoftmax, MobileClipImageEncoderExport


def test_classifier_with_softmax_output_shape_and_sums_to_one():
    classifier = MLP2Classifier(input_dim=512, num_classes=7)
    wrapped = ClassifierWithSoftmax(classifier)

    probabilities = wrapped(torch.randn(3, 512))

    assert probabilities.shape == (3, 7)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.all(probabilities >= 0)


def test_classifier_with_softmax_preserves_argmax():
    classifier = MLP2Classifier(input_dim=512, num_classes=7)
    classifier.eval()
    wrapped = ClassifierWithSoftmax(classifier)

    x = torch.randn(5, 512)
    with torch.no_grad():
        logits = classifier(x)
        probabilities = wrapped(x)

    assert torch.equal(logits.argmax(dim=-1), probabilities.argmax(dim=-1))


def test_mobileclip_image_encoder_export_is_unit_norm():
    import mobileclip

    model, _, _preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s0", pretrained=None
    )
    model.eval()

    wrapped = MobileClipImageEncoderExport(model)
    image = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        embedding = wrapped(image)

    assert embedding.shape == (1, 512)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(1), atol=1e-4)
