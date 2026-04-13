import torch
from soupsieve.css_types import pickle_register
from torchgen.api.functionalization import mutated_view_binding
import torch.nn as nn
class GeneCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size = 3):
        super().__init__()
        self.chn = chn
        #self.perc = torch.nn.Conv2d(chn , 3*(chn), 3, padding=1, padding_mode='circular', bias=False)
        self.w1 = torch.nn.Conv2d(chn + 3*(chn), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn-gene_size, 1, bias=False)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size

    def forward(self, x, update_rate=0.5):
        gene = x[:,-self.gene_size:,...]
        #y = self.perc(x)
        y = perception(x)
        #y = torch.cat((y,x), dim=1 )
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0, ).cuda() > 0.1
        #alive_mask = x[:, None, 3, ...]

        x = x[:,:x.shape[1]-self.gene_size,...] + y * update_mask * pre_life_mask


        #gene = self.dna_update(gene, alive_mask)
        x = torch.cat((x, gene), dim=1)
        return x


class GeneCA2(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size = 3):
        super().__init__()
        self.chn = chn
        #self.perc = torch.nn.Conv2d(chn , 3*(chn), 3, padding=1, padding_mode='circular', bias=False)
        self.w1 = torch.nn.Conv2d(chn + 3*(chn), hidden_n, 1)
        self.w3 = torch.nn.Conv2d(hidden_n, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn-gene_size, 1, bias=False)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size

    def forward(self, x, update_rate=0.5):
        gene = x[:,-self.gene_size:,...]
        #y = self.perc(x)
        y = perception(x)
        #y = torch.cat((y,x), dim=1 )
        y = torch.relu(self.w1(y))
        y = torch.relu(self.w3(y))
        y = self.w2(y)
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0, ).cuda() > 0.1
        #alive_mask = x[:, None, 3, ...]

        x = x[:,:x.shape[1]-self.gene_size,...] + y * update_mask * pre_life_mask


        #gene = self.dna_update(gene, alive_mask)
        x = torch.cat((x, gene), dim=1)
        return x

def perchannel_conv(x, filters):
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], mode='circular')
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")
ones = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda:0")
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device="cuda:0")
lap = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda:0")
gaus = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device="cuda:0")
moore_neighbors = torch.tensor([[1.0, 1.0, 1.0],
                                [1.0, 0.0, 1.0],
                                [1.0, 1.0, 1.0]], dtype=torch.float32, device="cuda:0"
                               )

von_neumann_neighbors = torch.tensor([[0.0, 1.0, 0.0],
                                      [1.0, 0.0, 1.0],
                                      [0.0, 1.0, 0.0]], dtype=torch.float32, device="cuda:0")

vertical_line_detector = torch.tensor([[0.0, 1.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [0.0, 1.0, 0.0]], dtype=torch.float32, device="cuda:0")

# 16. Horizontal Line Detector: Responds strongly to cells that are part of a 1-pixel thick horizontal line.
# Use: Differentiating lines from solid shapes.
horizontal_line_detector = torch.tensor([[0.0, 0.0, 0.0],
                                         [1.0, 1.0, 1.0],
                                         [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")



def perception(x):

    filters = torch.stack([sobel_x, sobel_x.T,lap])

    obs = perchannel_conv(x, filters)
    return torch.cat((x,obs), dim = 1 )


def gradnorm_perception(x):
  grad = perchannel_conv(x, torch.stack([sobel_x, sobel_x.T]))
  gx, gy = grad[:, ::2], grad[:, 1::2]
  state_lap = perchannel_conv(x, torch.stack([ident, lap]))
  return torch.cat([ state_lap, (gx*gx+gy*gy+1e-8).sqrt()], 1)


class IsoGeneCA(torch.nn.Module):
    def __init__(self, chn=12, hidden_n=96, gene_size = 3):
        super().__init__()
        self.chn = chn

        self.w1 = torch.nn.Conv2d(3*(chn), hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn-gene_size, 1, bias=False)
        self.w2.weight.data.zero_()
        self.gene_size = gene_size

    def forward(self, x, update_rate=0.5):
        gene = x[:,-self.gene_size:,...]
        #y = self.perc(x)
        y = gradnorm_perception(x)
        #y = torch.cat((y,x), dim=1 )
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device="cuda:0") + update_rate).floor()
        xmp = torch.nn.functional.pad(x[:, None, 3, ...], pad=[1, 1, 1, 1], mode="circular")
        pre_life_mask = torch.nn.functional.max_pool2d(xmp, 3, 1, 0, ).cuda() > 0.1
        #alive_mask = x[:, None, 3, ...]

        x = x[:,:x.shape[1]-self.gene_size,...] + y * update_mask * pre_life_mask


        #gene = self.dna_update(gene, alive_mask)
        x = torch.cat((x, gene), dim=1)
        return x