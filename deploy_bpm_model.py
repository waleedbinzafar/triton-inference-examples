# this script uses the triton client to deploy a model to triton server

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import sys

# the model is already saved in the model directory, but we need to prompt the server to load it
# Setting up client
triton_client = httpclient.InferenceServerClient(
            url="localhost:8000"
        )

model_name="bpm_librosa"

triton_client.load_model(model_name)

# check if the model is loaded
print(triton_client.is_model_ready(model_name=model_name))