# Use the official image as a parent image
FROM nvcr.io/nvidia/tritonserver:24.04-py3

# copy the requirements.txt file into the container
COPY ./requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# run the inference server in explicit mode
CMD ["tritonserver", "--model-repository=/models", "--model-control-mode=explicit"]