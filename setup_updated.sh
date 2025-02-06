conda create -n rppg-toolbox python=3.8 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rppg-toolbox

pip install -r requirements.txt
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
