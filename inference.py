import joblib
import os
import json
import numpy as np


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        return np.array([list(payload["instances"][0].values())])
    elif request_content_type == "text/csv":
        data = np.array([float(x) for x in request_body.split(",")])
        return data.reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    preds = model.predict(input_data)
    probs = model.predict_proba(input_data)[:, 1]
    return {"predictions": preds.tolist(), "probabilities": probs.tolist()}


def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
