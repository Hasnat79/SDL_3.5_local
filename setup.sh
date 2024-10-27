conda create -n sdl python=3.9 -y
conda activate sdl
pip install diffusers==0.31.0 transformers==4.46.0 accelerate==1.0.1
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install protobuf
pip install sentencepiece
