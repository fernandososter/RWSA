import torch
from sleep_rswa import SleepStagingRSWASystem
model=SleepStagingRSWASystem().eval(); b,t=2,8
with torch.no_grad(): out=model(torch.randn(b,t,4,900),torch.randn(b,t,1,300),torch.ones(b,t,dtype=torch.bool))
for k,v in out.items(): print(k,tuple(v.shape))
