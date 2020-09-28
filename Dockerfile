FROM nvcr.io/nvidia/pytorch:20.03-py3

COPY . retinanet/
RUN pip install --no-cache-dir -e retinanet/

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install albumentations
