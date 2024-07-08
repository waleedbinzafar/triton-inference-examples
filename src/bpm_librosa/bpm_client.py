# this script uses the triton client library to send a request to the bpm_librosa model
# the request is a .mp3 file that is read and sent to the model
# the model then calculates the bpm and returns it

import tritonclient.http as httpclient
import base64
import numpy as np
import librosa
import logging

if __name__=="__main__":
    client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
    audio_file = "audio.wav"
    with open(audio_file, "rb") as f:
        bytes_data = f.read()

    base64_bytes = base64.b64encode(bytes_data)
    base64_string = base64_bytes.decode("utf-8")

    model_name="bpm_librosa"

    inputs = []
    outputs = []
    
    input_name = "INPUT__0"
    output_name = "OUTPUT__0"
    
    numpy_data = np.asarray([base64_string], dtype=object)

    input_tensor = httpclient.InferInput(input_name, [1], "BYTES")
    output_tensor = httpclient.InferRequestedOutput(output_name)

    input_tensor.set_data_from_numpy(numpy_data.reshape([1]))
    
    inputs.append(input_tensor)
    outputs.append(output_tensor)

    response = client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs
    )

    output_data = response.as_numpy(output_name)
    print(output_data)