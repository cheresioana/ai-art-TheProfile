# Video description of the project

https://www.youtube.com/watch?v=RgU3i4qAVaI&t=5s


# ai art project


Deepfake is creating realistic forgeries that make people unable to distinguish reality from fiction, while filtering algorithms distort the perception by amplifying the interests of the user. From the way we perceive society to the way we perceive ourselves, the virtual environment is changing fundamental mechanisms of life. Due to the rapid growth of the virtual world and its hazy moral context, our society needs to build the groundwork of an ethical code for the online environment. Powerful technologies are used to produce art that captivates and impresses the general public. The artworks highlight ethical problems that our society is facing, creating a thoughtful space of analysis, experience and debate around the moral concerns in the technological era. This paper contains a short analysis on some of the main threats of artificial intelligence along with the proposal of art as a way to approach technology ethics on a large scale.

https://github.com/cheresioana/ai-art-TheProfile/blob/master/ArtPaper%20(4).pdf


https://github.com/cheresioana/ai-art-TheProfile/blob/master/The_role_of_art_in_shaping_artificial_intelligence_ethics_to_general_public.pdf

# Building blocks
This project was inspired and buit using the Openpose and Everybody Dance Now code.

# Installation

## Remote component
The remote component needs high computational power. The code was tested on a Linux machine with Python 2.7, NVIDIA Docker and 3 Tesla V100 GPUs.
For this project on the remote machine must be an image with a specific name (cheres-gan2). The docker container will be runned remotly by the local component, so there is no need to start it. For creating the image type:
```
docker build -t cheres-gan2 .
```

## Local component
The local component must be at the location of the installation, where it is in contact with the user. 
It needs a webcam and a screen.
The code was tested on a Windows machine with Python 3.7, CUDA, CUDNN and openpose library installed from source. 
For installing all the libraries:
```
pip3 install -r requirements.txt 
```
Change the file credentials.json with the correct data where the remote component will run. 
For starting the local component just type

```
python main.py
```


