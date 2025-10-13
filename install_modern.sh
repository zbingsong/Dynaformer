# must specify mkl version; pytorch cuda is based on old mkl, but conda automatically installs the latest mkl which is incompatible
conda install pytorch-geometric=2.6.1 pytorch pytorch-cuda cuda-version=12.4 mkl=2023.0.0 -c rusty1s -c nvidia -c pytorch -c conda-forge -y
conda install pymol rdkit openbabel joblib -c schrodinger -c conda-forge -y
