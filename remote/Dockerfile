FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt-get update -y && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update -y && apt-get install -y \
 python3.7 \
 python3-pip \
 curl \
 git

RUN pip3 install --upgrade pip

RUN mkdir -p /home/cheres

WORKDIR /home/cheres

COPY  . .

RUN pip3 install -r requirements.txt


CMD ["python3", "main.py"]

