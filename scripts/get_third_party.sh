#!/usr/bin/env bash

cd $(dirname $0)/..

mkdir -p third_party && cd third_party

THIRDPARTYLOCAL="local"
SENTENCEPIECEURI="https://github.com/google/sentencepiece.git"
LIBTORCHURI="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.3.1%2Bcpu.zip"
BOOSTURI="https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz"
NLOHMANNURI="https://github.com/nlohmann/json.git"

echo "Installing sentencepiece, make sure you've installed the requirements, which on Ubuntu is:"
echo "sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev"

git clone $SENTENCEPIECEURI
cd sentencepiece
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$(realpath ../../${THIRDPARTYLOCAL}) ..
make -j $(nproc --ignore=1)
make install
# cleaning up
cd ../..
rm -Rf sentencepiece


echo "installing cpu-libtorch, if you want the CUDA version it's available at https://pytorch.org"
wget $LIBTORCHURI -O libtorch.zip
unzip libtorch.zip
rm libtorch.zip

echo "installing nlohmann json..."
git clone ${NLOHMANNURI}
cd json
mkdir build && cd build
cmake -DJSON_BuildTests=OFF -DCMAKE_INSTALL_PREFIX=$(realpath ../../${THIRDPARTYLOCAL}) ..
make install
cd ..


#echo "installing boost"
#wget $BOOSTURI -O boost.tar.gz
#tar xzf boost.tar.gz
#rm boost.tar.gz

