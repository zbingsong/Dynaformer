# conda install numpy=1.22.4 pytorch=2.0.1=py3.9_cuda11.8_cudnn8.7.0_0 torchvision=0.15.2=py39_cu118 torchaudio=2.0.2=py39_cu118 pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge -y
# conda install cuda-version=11.8 -c nvidia -y

# pip install "pip<24" "pybind11[global]" Cython==3.0.11 ninja lmdb
# pip install torch-scatter==2.1.1 torch-sparse==0.6.17 -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
# pip install dgl==0.8.0 -f https://data.dgl.ai/wheels/repo.html

# pip install torch-geometric==2.0.4 tensorboardX==2.6 ogb==1.3.6 rdkit-pypi==2022.9.4 \
#             icecream torchmetrics==1.0.0 tensorboard setuptools==59.5.0 \
#             protobuf==3.8.0 wandb scipy pandas
conda install numpy==1.22.4

cd fairseq
pip install .
python setup.py build_ext --inplace
cd ..
# python setup_cython.py build_ext --inplace
