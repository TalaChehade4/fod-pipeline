"""On-device export utilities.

Converts the pipeline's PyTorch models (YOLO detector, MobileCLIP image
encoder, MLP classifier) into the native formats mobile runtimes expect:

    - ONNX (``.onnx``), loaded on-device via ONNX Runtime Mobile, for Android.
    - Core ML (``.mlpackage``) for iOS.

See ``fod_pipeline.mobile.export_all`` for the end-to-end CLI, or the
per-model ``export_*`` modules to convert a single model.
"""
