docker run --gpus all -it --rm ^
    -p8000:8000 ^
    -p8001:8001 ^
    -p8002:8002 ^
    -v %cd%/model_repository:/models ^
    nvcr.io/nvidia/tritonserver:24.02-py3 ^
    bash -c "ls -l /opt/tritonserver/backends && tritonserver --model-repository=/models --log-verbose=1" 