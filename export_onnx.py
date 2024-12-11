from models.ChangeRD import *
import torch

def init():
    net = ChangeRD()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('checkpoints/off_CD_ChangeRD_LEVIR_b8_lr0.0001_adamw_train_test_400_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim111_256/best_ckpt.pt', map_location=device)
    saved_weights = checkpoint['model_G_state_dict']
    new_state_dict = {}
    for k, v in saved_weights.items():
        if k.startswith('module.'):
            name = k[7:]  # remove the "module." prefix
        else:
            name = k
        new_state_dict[name] = v

    # create a new model and load the new state dict
    net.load_state_dict(new_state_dict)
    net = net.eval()
    return net

def export_onnx():
    net = init()
    # 创建两个虚拟输入，每个都是 1x3x512x512
    dummy_input1 = torch.randn(1, 3, 512, 512)
    dummy_input2 = torch.randn(1, 3, 512, 512)
    
    # 将两个输入组合成一个元组
    dummy_input = (dummy_input1, dummy_input2)
    
    input_names = ['input1', 'input2']
    output_names = ['output']
    
    torch.onnx.export(net, dummy_input, 'levir.onnx', verbose=True, 
                      input_names=input_names, output_names=output_names,
                      opset_version=18)

if __name__ == '__main__':
    export_onnx()
