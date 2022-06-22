curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo docker pull ianlienfa/nsd:hw3.0
sudo docker run -v ~/BBGym-RL:/BBGym -it ianlienfa/nsd:hw3.0 /bin/bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip && unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip
cd build && cmake ..