"""Enable RAPIDS before importing pandas or scikit-learn.

USEDCAR_DEVICE=auto (default), cuda (require GPU), or cpu.
Restart the notebook kernel when changing this setting.
"""
import os
import sys

_STATE = None


def enable_gpu():
    global _STATE
    if _STATE is not None:
        return _STATE
    mode = os.environ.get("USEDCAR_DEVICE", "auto").lower()
    if mode not in {"auto", "cuda", "cpu"}:
        raise ValueError("USEDCAR_DEVICE must be auto, cuda, or cpu")
    if mode == "cpu":
        _STATE = False
        print("UsedCarAI: CPU mode")
        return False
    if "sklearn" in sys.modules or "pandas" in sys.modules:
        raise RuntimeError("Restart the kernel and run the GPU setup cell first (before pandas/sklearn).")
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device found")
        cp.arange(4).sum().item()
        import importlib.util
        if importlib.util.find_spec("cuml") is None or importlib.util.find_spec("cudf") is None:
            raise ImportError("RAPIDS cuML/cuDF is not installed")
    except (ImportError, RuntimeError) as exc:
        if mode == "cuda":
            raise RuntimeError("GPU setup failed. Install requirements-gpu.txt in this kernel.") from exc
        _STATE = False
        print(f"UsedCarAI: CPU fallback ({exc}). Install requirements-gpu.txt for GPU mode.")
        return False
    # Fail explicitly if installation fails midway: do not silently mix backends.
    import cudf.pandas
    cudf.pandas.install()
    import cuml.accel
    cuml.accel.install(log_level=os.environ.get("CUML_ACCEL_LOG_LEVEL", "info"))
    _STATE = True
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(name, bytes):
        name = name.decode()
    print(f"UsedCarAI: RAPIDS enabled on {name}; unsupported operations use CPU")
    return True


def load_model(path):
    """Load a pipeline and convert existing CPU estimators for GPU inference.

    Conversion is in memory only; the saved model and learned weights are unchanged.
    """
    import joblib
    pipeline = joblib.load(path)
    if not _STATE:
        return pipeline
    from cuml.accel import is_proxy
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LinearRegression
    model = pipeline.named_steps["model"]
    if not is_proxy(model):
        classes = {
            "RandomForestRegressor": RandomForestRegressor,
            "LinearRegression": LinearRegression,
        }
        cls = classes.get(type(model).__name__)
        if cls is None:
            raise TypeError(f"No GPU conversion configured for {type(model).__name__}")
        gpu_model = cls.from_sklearn(model)
        gpu_model.set_params(output_type="numpy")
        pipeline.steps[-1] = (pipeline.steps[-1][0], gpu_model)
        print(f"UsedCarAI: converted saved {type(model).__name__} for GPU inference")
    # Old CPU exports use sparse one-hot matrices; GPU RF requires dense input.
    pre = pipeline.named_steps["preprocessor"]
    for _, transformer, _ in pre.transformers_:
        for _, step in getattr(transformer, "steps", []):
            if hasattr(step, "sparse_output"):
                step.set_params(sparse_output=False)
    pre.sparse_output_ = False
    return pipeline
