# this script uses the triton client library to send a request to the bpm_librosa model
# the request is a .mp3 file that is read and sent to the model
# the model then calculates the bpm and returns it

import tritonclient.http as httpclient
import numpy as np
import librosa
import logging

if __name__=="__main__":
    client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    audio_file = "audio.wav"
    y, sr = librosa.load(audio_file, sr=16000)
    input_data = np.array(y, dtype=np.float32)
    input_data = input_data.tobytes()
    input_data = httpclient.InferInput("INPUT__0", [1, len(input_data)])
    input_data.set_data_from_bytes(input_data)
    output_data = httpclient.InferRequestedOutput("OUTPUT__0")
    response = client.infer(model_name="bpm_librosa", inputs=[input_data], outputs=[output_data])
    bpm = response.as_numpy("OUTPUT__0")
    print(f"bpm: {bpm}")
