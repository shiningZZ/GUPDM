import numpy as np
from torchstat import stat
from torchsummary import summary
from mymodel_channelattention_lut3 import UNet
from thop import profile, clever_format
model = UNet()
# stat(model, (3, 256, 256))
summary(model.cuda(), input_size=(1, 3, 256, 256), batch_size=1)
# flops, para = profile(model, inputs=((1, 3, 256, 356), (1, 3, 256, 256), (1, 3, 256, 256)))
# print(flops)
# print(para)
# print(np.sum([p.nelement() for p in model.paramets ers()]).item())  # 49451995
