FROM python:3.8.12

WORKDIR /root
RUN apt update && apt upgrade -y
RUN apt install -y git vim
RUN apt install -y libgl1-mesa-dev
RUN python -m pip install --upgrade pip
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" /dev/null
RUN git clone https://github.com/saityyy/generate_fractal.git

WORKDIR /root/generate_fractal/
RUN pip install -r requirements.txt
#RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html