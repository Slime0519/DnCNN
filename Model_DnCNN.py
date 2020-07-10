import torch.nn as nn
import torch.nn.init as init
import math

class DnCNN(nn.Module):
    def __init__(self,iscolor = 'False',depth = 17):
        super(DnCNN, self).__init__()

        #def type 1(first) layer
        if(iscolor == 'False'):
            color_channel = 1
        else:
            color_channel = 3
        self.type1_conv = nn.Conv2d(in_channels=color_channel,out_channels=64,kernel_size=3,padding=1, padding_mode='zeros',bias=True)

        #def type2 layer
        self.type2_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode= 'zeros',bias=True)
        self.type2_bn = nn.BatchNorm2d(64,eps = 0.0001, momentum = 0.95)

        #def type3 layer
        self.type3_conv = nn.Conv2d(in_channels=64,out_channels=color_channel, kernel_size=3 ,padding=1, padding_mode='zeros',bias=True)

        self.dncnn = nn.Sequential()
        self.dncnn.add_module("patch extraction", self.type1_conv)
        self.dncnn.add_module("relu",nn.ReLU(inplace=True))
        for i in range(depth-2):
            self.dncnn.add_module("{}th conv in 2nd part".format(i),self.type2_conv)
            self.dncnn.add_module("{}th bn in 2nd part".format(i),self.type2_bn)
            self.dncnn.add_module("{}th ReLU in 2nd part".format(i),nn.ReLU(inplace=True))

        self.dncnn.add_module("last conv",self.type3_conv)
        self._initialize_weights()

    def forward(self, x):
        y = x
     #   print(self.dncnn)
        out = self.dncnn(y)
        return y-out

    def _initialize_weights(self):
        print(self.dncnn)
        for m in self.dncnn.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal_(m.weight)
                init.kaiming_normal_(m.weight, a=0,mode='fan_in')
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
               # init.constant_(m.weight, 1)
              #  init.constant_(m.bias, 0)
