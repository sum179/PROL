
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers.weight_init import trunc_normal_
from timm.models.layers import Mlp
from copy import deepcopy

class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        print("Nb classes: ",nb_classes)
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=True))
        # single_head.append(SingleTaskHead(embed_dim, nb_classes))
        # single_head.append(nn.Linear(embed_dim, nb_classes, bias=False))
        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 
                    # nn.init.uniform_(m.bias, -1, 1)


    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)


    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim).cuda())
        _fc = nn.Linear(self.embed_dim, nb_classes, bias=True).cuda()
        # _fc = SingleTaskHead(self.embed_dim, nb_classes).cuda()
        # _fc = nn.Linear(self.embed_dim, nb_classes, bias=False).cuda()
        
        trunc_normal_(_fc.weight, std=.02)
        nn.init.constant_(_fc.bias, 0)
        
        # nn.init.uniform_(single_head[0].weight, -1, 1) 

        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)
   

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad=False

        self.heads.append(new_head)
        
        # for p in self.heads.parameters():
        #     # p.requires_grad=False
        #     print(p.requires_grad, end=" ")
        # print("\n == print head param complete ==")

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(self.heads[ti](fc_inp))
        out = {'logits': torch.cat(out, dim=1)}
        return out


    def forward2(self, x, max_t):
        out = []
        for ti in range(max_t):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(self.heads[ti](fc_inp))
        out = {'logits': torch.cat(out, dim=1)}
        return out


class SingleTaskHead(nn.Module):
    def __init__(self,embed_dim, nb_classes,factor=12):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim/factor)),
            # nn.ReLU(),
            nn.Linear(int(embed_dim/factor), nb_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
