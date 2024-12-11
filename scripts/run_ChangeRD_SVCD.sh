#!/usr/bin/env bash

#GPUs
gpus=0,1,2,3

#Set paths
checkpoint_root=/data/jingwei/change_detection_RD/ChangeRDv3/checkpoints
vis_root=/data/jingwei/change_detection_RD/ChangeRDv3/vis
data_name=SVCD


img_size=256    
batch_size=32   
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
project_name=off1_CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim1_${embed_dim}

python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --pretrain ${pretrain} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}
