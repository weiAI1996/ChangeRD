gpus=1
data_name=LEVIR
net_G=ChangeRD #This is the best version
split=test
vis_root=/data/jingwei/change_detection_RD/ChangeRDv3/vis
project_name=off_CD_ChangeRD_LEVIR_b8_lr0.0001_adamw_train_test_400_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim111_256
checkpoints_root=/data/jingwei/change_detection_RD/ChangeRDv3/checkpoints
checkpoint_name=best_ckpt.pt
img_size=512
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

python eval_cd.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


