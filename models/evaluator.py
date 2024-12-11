import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)
        # for i in range(1000):
        #     # 创建输入网络的tensor
        #     # from fvcore.nn import FlopCountAnalysis, parameter_count_table

        #     # tensor = (torch.rand(1, 3, 512, 512).float().cuda(),torch.rand(1, 3, 512, 512).float().cuda())

        #     # # 分析FLOPs
        #     # flops = FlopCountAnalysis(self.net_G, tensor)
        #     # print("FLOPs: ", flops.total())
        #     # from ptflops import get_model_complexity_info
        #     # macs, params = get_model_complexity_info(self.net_G , (3, 512, 512), as_strings=True,
        #     #                                     print_per_layer_stat=True, verbose=True)
        #     # print(macs)
        #     # print(params)


        #     from thop import profile
        #     input = torch.randn(1, 3, 512, 512).float().cuda()
        #     flops, params = profile(self.net_G, inputs=(input, input,))
            
        #     from thop import clever_format
        #     flops, params = clever_format([flops, params], "%.3f")
        #     print('flops:{}'.format(flops))
        #     print('params:{}'.format(params))
        #     print(f"Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #     print(f"Max memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.Alinged_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            saved_weights = checkpoint['model_G_state_dict']
            new_state_dict = {}
            for k, v in saved_weights.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove the "module." prefix
                else :
                    name = k
                new_state_dict[name] = v

            # create a new model and load the new state dict
            self.net_G.load_state_dict(new_state_dict)

            self.net_G.to(self.device)
            # from thop import profile
            # input = torch.randn(1, 3, 512, 512).float().cuda()
            # flops, params = profile(self.net_G, inputs=(input, input,))
            # print('flops:{}'.format(flops))
            # print('params:{}'.format(params))
            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 1) == 0:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 1) == 0:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input_aligned = utils.make_numpy_grid(de_norm(self.Alinged_pred))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input_aligned, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            mask_path = os.path.join(self.vis_dir,'mask')
            alined_path = os.path.join(self.vis_dir,'alined')
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)
            if not os.path.exists(alined_path):
                os.makedirs(alined_path)
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
            mask_file_name = os.path.join(
                mask_path, 'eval_' + str(self.batch_id)+'.png')
            alined_file_name = os.path.join(
                alined_path, 'eval_' + str(self.batch_id)+'.jpg')
            # print(self._visualize_pred().shape)
            plt.imsave(file_name, vis)
            cv2.imwrite(mask_file_name, self._visualize_pred().squeeze().cpu().numpy())
            plt.imsave(alined_file_name, vis_input_aligned)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        pred = self.net_G(img_in1, img_in2)
        self.G_pred = pred[0][-1]
        self.Alinged_pred = pred[-1]
        # print(self.Alinged_pred.size())

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
