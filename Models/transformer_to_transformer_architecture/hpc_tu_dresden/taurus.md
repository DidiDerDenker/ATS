ssh davo557d@taurus.hrsk.tu-dresden.de
srun --pty -p ml -n 1 -c 4 --gres=gpu:1 --mem-per-cpu 5772 -t 08:00:00 bash -l 

module load PythonAnaconda
mkdir transformer-kernel
conda create --prefix /home/davo557d/user-kernel/transformer-kernel python=3.6
source activate /home/davo557d/user-kernel/transformer-kernel
conda install -c anaconda ipykernel
python -m ipykernel install --user --name transformer-kernel --display-name="transformer-kernel"
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access/linux-ppc64le/
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install tokenizers regex
git clone https://github.com/huggingface/transformers
cd transformers
pip install . --user
