#  this script is the model.py file for the bpm_librosa model
#  it contains the model class and the necessary functions to load the model
#  and make predictions
#  the model is a simple function that takes a .mp3 file as input and returns the bpm

import librosa
import base64
import numpy as np
import triton_python_backend_utils as pb_utils
import logging
import io

class TritonPythonModel:
    def initialize(self, args):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                input_data = pb_utils.get_input_tensor_by_name(request, "INPUT__0")
                numpy_data = input_data.as_numpy()[0].decode('utf-8')
                decoded_data = base64.b64decode(numpy_data)
                io_object = io.BytesIO(decoded_data)
                
                bpm = self.get_bpm(io_object)
                
                output_data = np.array([bpm], dtype=np.float32)
                
                output_tensor = pb_utils.Tensor("OUTPUT__0", output_data)
                
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(inference_response)
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                response = pb_utils.InferenceResponse(output_tensors=[], error="Error processing request")
                responses.append(response)
        return responses

    def get_bpm(self, audio_data):
        y,sr = librosa.load(audio_data)
        # y = audio_data
        # sr= 16000
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        return tempo

    def finalize(self):
        self.logger.info("Model finalized")