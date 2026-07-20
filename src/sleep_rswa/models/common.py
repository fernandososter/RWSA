import math
import torch
import torch.nn as nn

def make_group_norm(channels, max_groups=8):
    for groups in range(min(max_groups, channels),0,-1):
        if channels % groups == 0: return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1,channels)

class SEBlock(nn.Module):
    def __init__(self,n_ch,r=8):
        super().__init__(); mid=max(1,n_ch//r)
        self.fc=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(n_ch,mid,bias=False),nn.ReLU(inplace=True),nn.Linear(mid,n_ch,bias=False),nn.Sigmoid())
    def forward(self,x): return x*self.fc(x).unsqueeze(-1)

class MultiKernelCNNBranch(nn.Module):
    def __init__(self,in_ch,out_ch,kernels,n_layers=4,drop=0.35):
        super().__init__(); per=out_ch//len(kernels); rem=out_ch-per*len(kernels); paths=[]
        for i,k in enumerate(kernels):
            co=per+(rem if i==0 else 0); layers=[]; ci=in_ch
            for _ in range(n_layers):
                layers += [nn.Conv1d(ci,co,k,padding=k//2,bias=False),make_group_norm(co),nn.ReLU(inplace=True),nn.MaxPool1d(2,2)]; ci=co
            paths.append(nn.Sequential(*layers))
        self.paths=nn.ModuleList(paths); self.drop=nn.Dropout(drop)
    def forward(self,x):
        ys=[p(x) for p in self.paths]; m=min(y.shape[-1] for y in ys)
        return self.drop(torch.cat([y[...,:m] for y in ys],1))
