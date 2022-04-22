import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class netG(nn.Module):
    def __init__(self, layer_sizes, latent_size, attSize):
        super().__init__()
        self.MLP = nn.Sequential()
        input_size = latent_size + attSize
        for i, (in_size, out_size) in enumerate( zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A%i"%(i), module=nn.LeakyReLU(0.2, True))
            else:
                self.MLP.add_module(name="ReLU", module=nn.ReLU(inplace=True))
        self.apply(weights_init)
    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x
class netD(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)
    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h
class netP(nn.Module):
    def __init__(self, layer_sizes,att_size):
        super().__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate( zip([att_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes) :
                self.MLP.add_module(name="A%i"%(i), module=nn.LeakyReLU(0.2, True))
        self.apply(weights_init)
    def forward(self, x, c=None):
        x = self.MLP(x)
        return x