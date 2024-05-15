# #!/bin/bash

#conda create -n topox python=3.11.3
#conda activate topox

pip install --upgrade pip

CUDA="cu115" # if available, select the CUDA version suitable for your system
             # e.g. cpu, cu102, cu111, cu113, cu115
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html

pip install -e '.[all]'

pip install git+https://github.com/pyt-team/TopoNetX.git
pip install git+https://github.com/pyt-team/TopoModelX.git
pip install git+https://github.com/pyt-team/TopoEmbedX.git

pytest

pre-commit install
