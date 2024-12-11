#!/usr/bin/env bash

#GPUs
gpus=2,3

#Set paths
checkpoint_root=/data/jw/change_detection/ChangeRDv3/checkpoints
vis_root=/data/jw/change_detection/ChangeRDv3/vis
data_name=LEVIR


img_size=512    
batch_size=2   
lr=0.0001         
max_epochs=400
embed_dim=256

net_G=ChangeRD      
lr_policy=linear
optimizer=adamw                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

#Initializing from pretrained weights
pretrain=pretrained/backbone_weights.pth

#Train and Validation splits
split=train         #trainval
split_val=test      #test
project_name=debug

python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --pretrain ${pretrain} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}
