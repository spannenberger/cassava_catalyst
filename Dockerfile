FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
LABEL Name=cv Version=0.0.1

RUN apt update -y
RUN apt install -y git
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt -U
COPY . /workspace/cv/
WORKDIR /workspace/cv/
RUN rm -rf /workspace/requirements.txt
RUN apt-get -y update && apt-get install tmux -y