conda create -n rppg-toolbox python=3.8 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rppg-toolbox

pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
