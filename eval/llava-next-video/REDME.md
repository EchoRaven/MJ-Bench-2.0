# conda environment

```
git clone git@github.com:EchoRaven/MJ-Bench-2.0.git
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install boto3
cd aws_upload
bash download.sh
# 下载数据集
cd ../eval/llava-next-video
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e ".[train]"
cd ../
```
# run this code

```
sbatch bash_7B.sh
sbatch bash_34B.sh
```
