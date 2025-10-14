conda install protobuf=3.20.3 -c conda-forge -y
conda install numpy=1.22.4 pytorch=2.1.0=py3.9_cuda11.8_cudnn8.7.0_0 torchvision=0.16.0=py39_cu118 torchaudio=2.1.0=py39_cu118 \
	pytorch-cuda=11.8 cuda-version=11.8 torchdata=0.7.0 \
	-c pytorch -c nvidia -c conda-forge -y
conda install pytorch-scatter=2.1.2=py39_torch_2.1.0_cu118 pytorch-sparse=0.6.18=py39_torch_2.1.0_cu118 -c pyg -c conda-forge -y
conda install pytorch-geometric=2.4.0=py39_torch_2.1.0_cu118 -c rusty1s -c conda-forge -y
conda install cython=3.0.11 tensorboardx=2.6.2 tensorboard=2.10.0 -c conda-forge -y
conda install dglteam/label/cu118::dgl -y
conda install pymol -c schrodinger -c conda-forge -y
conda install torchmetrics ninja python-lmdb pybind11-global ogb icecream wandb pandas rdkit openbabel -c conda-forge -y
conda install pip=24.0 -y
cd fairseq
pip install --editable ./
#python setup.py build_ext --inplace
cd ..
python setup_cython.py build_ext --inplace
