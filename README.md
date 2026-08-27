# finalproject
Resource-Aware and Adaptive Data Partitioning for Distributed Deep Neural Network Training
Manual and Resource-Aware and Adaptive Data Partitioning for Distributed Deep Neural Network Training
# Dataset

The experiments use the Tiny ImageNet-200 dataset.
python environment should at least include:
fastapi
uvicorn
torch
torchvision
psutil
numpy

The dataset is not redistributed in this repository.
Users should obtain the dataset first
the **.py are experiment implementation
the **.csv is the measurement results of indicators
the **.png is the result visual

1.First "bash download_dataset.sh" to get the dataset on host machine

2.Then on your worker machine:
cd ~/your files
source ~/<your_python_virtual_environment>/bin/activate

unset CPU_LIMIT
export WORKER_ROOT=~/dist_resnet_worker
uvicorn worker:app --host 0.0.0.0 --port 8000

to start your worker machine
3.Get three workers ip address

4.Then about manual method:
cd ~/your files
source ~/<your_python_virtual_environment>/bin/activate

python coordinator_manual_weight.py \
  --data-root (The actual path where the Tiny ImageNet-200 training set is located on the computer) \
  --workers B=http://ip:8000 C=http://ip:8000 D=http://ip:8000 \
  --profile-csv ./adaptive_auto_work_gpu/adaptive_auto_metrics.csv \
  --profile-round latest \
  --weights cpu_util=<set any value> gpu_util=<set any value> throughput=<set any value> memory_headroom<set any value> time=<set any value> \
  --previous-ratios B=33.33 C=33.33 D=33.34 \
  --smoothing 1.0 \
  --round-id 201 \
  --work-dir ./manual_weight_auto_style_tests \
  --model-name resnet18 \
  --batch-size 64 \
  --lr 0.01 \
  --local-epochs 2 \
  --cleanup-workers

5. The auto method:
cd ~/your files
source ~/<your_python_virtual_environment>/bin/activate

python coordinator_adaptive_auto or coordinator_adaptive_auto(2,3).py \
  --data-root /uolstore/home/users/drgn0194/Downloads/tiny-imagenet-200/train \    (The actual path where the Tiny ImageNet-200 training set is located on the computer)
  --workers B=http://ip:8000 C=http://ip:8000 D=http://ip:8000 \
   --initial-ratios B=33.33 C=33.33 D=33.34 \
  --work-dir ./adaptive_auto_work_gpu \
  --model-name resnet18 \
  --batch-size 64 \
  --lr 0.01 \
  --local-epochs 2 \
  --target-gap-seconds 6 \
  --max-rounds 10 \
  --smoothing 0.70 \
  --cleanup-workers
