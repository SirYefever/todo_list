FROM nvcr.io/nvidia/tritonserver:22.12-py3

WORKDIR /app
COPY model_repository /models

# If additional dependencies are required, uncomment below
# COPY requirements.txt .
# RUN pip install -r requirements.txt

CMD ["tritonserver", "--model-repository=/models"] 