import numpy as np
import pandas as pd

from edgebettor_ml.modeling.datasets import PreprocessArtifacts, transform_with
from edgebettor_ml.modeling.model_nn import MultiHeadNet


def test_infer_output_ranges():
    # minimal sanity test on model output range 0..1
    model = MultiHeadNet(input_dim=4)
    X = np.zeros((2, 4), dtype=float)
    with np.errstate(all="ignore"):
        out = model(
            __import__("torch").tensor(X, dtype=__import__("torch").float32)
        )
    for k, v in out.items():
        arr = v.detach().numpy()
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0)


