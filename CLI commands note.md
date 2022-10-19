### All TPU ml code tutorials:
https://cloud-dot-devsite-v2-prod.appspot.com/tpu/docs/tutorials

### Profiling for TPU memory usage etc: 
https://cloud.google.com/tpu/docs/profile-tpu-vm


### Create TPU VM helpers:
```
PROJECT_ID=sunny-avocado
gsutil mb -c standard -l us-central1 gs://tpu-train-test-auv

TPU_NAME=auv-tpu-vm-3

gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=us-central1-b \
  --accelerator-type=v2-8 \
  --version=tpu-vm-tf-2.10.0 \
  --network=vpc-network-1 \
  --subnetwork=vpc-network-1
  
gcloud compute firewall-rules create --network=vpc-network-1 allow-ssh --allow=tcp:22

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=us-central1-b
```

### Run demo training in the TPU VM
```
pip3 install -r /usr/share/tpu/models/official/requirements.txt 
pip3 install --user tensorflow-model-optimization>=0.1.3

export PYTHONPATH="/usr/share/tpu/models:${PYTHONPATH}"
export STORAGE_BUCKET=gs://tpu-train-test-auv
export MODEL_DIR=${STORAGE_BUCKET}/resnet-2x
export DATA_DIR=gs://cloud-tpu-test-datasets/fake_imagenet
export IMAGENET_DIR=gs://cloud-tpu-test-datasets/fake_imagenet
export TPU_NAME=local

cd /usr/share/tpu/models

python3 official/vision/train.py \
--experiment=resnet_rs_imagenet \
--mode=train_and_eval \
--model_dir=$MODEL_DIR \
--tpu=$TPU_NAME \
--config_file=official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs50_i160.yaml \
--params_override="task.train_data.input_path=$IMAGENET_DIR/train*, task.validation_data.input_path=$IMAGENET_DIR/valid*, trainer.train_steps=100"
```