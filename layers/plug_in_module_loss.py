import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


# # 模型结构
# args.use_fpn      # 是否使用特征金字塔
# args.use_selection # 是否使用选择器
# args.use_combiner  # 是否使用组合器
#
# # 损失权重
# args.lambda_s      # 选中区域损失权重
# args.lambda_n      # 丢弃区域损失权重
# args.lambda_b      # FPN基础损失权重
# args.lambda_c      # 组合器损失权重
# use_selection,lambda_s,num_classes,lambda_n,batch_size,device,lambda_c

class PIMLoss(nn.Module):
    def __init__(self, cfg: dict, num_class: int):
        super(PIMLoss, self).__init__()
        # self.cfg = {'num_class': num_class}
        # self.cfg.update({'use_fpn': cfg.PLUG_MODEL.USE_FPN})
        # self.cfg.update({'use_selection': cfg.PLUG_MODEL.USE_SELECTION})
        # self.cfg.update({'use_combiner': cfg.PLUG_MODEL.USE_COMBINER})
        # self.cfg.update({'lambda_s': cfg.PLUG_MODEL.LAMBDA_S})
        # self.cfg.update({'lambda_n': cfg.PLUG_MODEL.LAMBDA_N})
        # self.cfg.update({'lambda_b': cfg.PLUG_MODEL.LAMBDA_B})
        # self.cfg.update({'lambda_c': cfg.PLUG_MODEL.LAMBDA_C})
        # print()
        # self.args = pim_config
        self.cfg = SimpleNamespace()
        self.cfg.num_classes = num_class
        self.cfg.use_fpn = cfg.PLUG_MODEL.USE_FPN
        self.cfg.use_selection = cfg.PLUG_MODEL.USE_SELECTION
        self.cfg.use_combiner = cfg.PLUG_MODEL.USE_COMBINER
        self.cfg.lambda_s = cfg.PLUG_MODEL.LAMBDA_S
        self.cfg.lambda_n = cfg.PLUG_MODEL.LAMBDA_N
        self.cfg.lambda_b = cfg.PLUG_MODEL.LAMBDA_B
        self.cfg.lambda_c = cfg.PLUG_MODEL.LAMBDA_C

    def forward(self, outs, labels):
        # args = self.args
        loss = 0.
        batch_size = outs['layer1'].shape[0]
        device = outs['layer1'].device
        for name in outs:
            if "select_" in name:
                if not self.cfg.use_selection:
                    raise ValueError("Selector not use here.")
                if self.cfg.lambda_s != 0:
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, self.cfg.num_classes).contiguous()
                    loss_s = nn.CrossEntropyLoss()(logit, labels.unsqueeze(1).repeat(1, S).flatten(0))
                    loss += self.cfg.lambda_s * loss_s
                else:
                    loss_s = 0.0

            elif "drop_" in name:
                if not self.cfg.use_selection:
                    raise ValueError("Selector not use here.")

                if self.cfg.lambda_n != 0:
                    S = outs[name].size(1)
                    logit = outs[name].view(-1, self.cfg.num_classes).contiguous()
                    n_preds = nn.Tanh()(logit)
                    # labels_0 = torch.zeros([self.cfg.batch_size * S, self.cfg.num_classes]) - 1
                    labels_0 = torch.zeros([batch_size * S, self.cfg.num_classes]) - 1
                    # labels_0 = labels_0.to(self.cfg.device)
                    labels_0 = labels_0.to(device)
                    loss_n = nn.MSELoss()(n_preds, labels_0)
                    loss += self.cfg.lambda_n * loss_n
                else:
                    loss_n = 0.0

            elif "layer" in name:
                if not self.cfg.use_fpn:
                    raise ValueError("FPN not use here.")
                if self.cfg.lambda_b != 0:
                    # here using 'layer1'~'layer4' is default setting, you can change to your own
                    loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                    loss += self.cfg.lambda_b * loss_b
                else:
                    loss_b = 0.0

            elif "comb_outs" in name:
                if not self.cfg.use_combiner:
                    raise ValueError("Combiner not use here.")

                if self.cfg.lambda_c != 0:
                    loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                    loss += self.cfg.lambda_c * loss_c

            elif "ori_out" in name:
                loss_ori = F.cross_entropy(outs[name], labels)
                loss += loss_ori
        return loss
