# this script uses the triton client library to send a request to the bpm_librosa model
# the request is a .mp3 file that is read and sent to the model
# the model then calculates the bpm and returns it

import tritonclient.http as httpclient
import numpy as np
import librosa
import logging

if __name__=="__main__":
    client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
    audio_file = "audio.wav"
    y, sr = librosa.load(audio_file, sr=16000)

    model_name="bpm_librosa"

    inputs = []
    outputs = []
    
    input_name = "INPUT__0"
    output_name = "OUTPUT__0"
    audio_data = y
    # audio_data = np.expand_dims(audio_data, axis=0)

    inputs.append(httpclient.InferInput(input_name, audio_data.shape, "FP32"))
    outputs.append(httpclient.InferRequestedOutput(output_name))

    inputs[0].set_data_from_numpy(audio_data)
    response = client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs
    )

    output_data = response.as_numpy(output_name)
    print(output_data)