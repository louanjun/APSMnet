import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.jit.annotations import Optional, Tuple

def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 #if downsample == False else 2 改
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Swish_act(nn.Module):
    def __init__(self):
        super(Swish_act, self).__init__()
 
    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 #if self.downsample == False else 2    改
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(1, 1, 0)  #改(3, 2, 1)————（1, 1, 0） 改
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                #Swish_act(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                #Swish_act(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                #Swish_act(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
   
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)  

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        
        #print("x:",x.shape)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)     
        
        #print("q:",q.shape)  
        #print("k:",k.shape)  

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
       
        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
       
        #print("dots:",dots.shape)
        #print("relative_bias:",relative_bias.shape)

        dots = dots + relative_bias
        attn = self.attend(dots)
        
        #print("attn_size:",attn.shape)
        #print("v_size:",v.shape)
        
        #v0 = torch.zeros(attn.shape[0],self.heads,attn.shape[2],32)
        #for i in range(1,attn.shape[2]):
        #    v0[:,:,i,:] = v[:,:,0,:]
        v = v.expand(attn.shape[0],-1,-1,-1) 
        v = v.expand(-1,-1,attn.shape[2],-1) 
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        
        if self.downsample:
            #print("x1:",x.shape)
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            #print("x2:",x.shape)
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class InceptionAux(nn.Module):                                    #定义辅助分类器
    def __init__(self, ih, num_classes,channels):
        super(InceptionAux, self).__init__()
        #self.pool = nn.AvgPool2d(ih , 1)  #改 // 32
        self.pool = nn.AvgPool2d(ih , 1)  #改 // 32
        self.fc = nn.Linear(channels, num_classes, bias=False)

    def forward(self, x):                                         #定义正向传播过程
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=6, block_types=['C', 'C', 'T', 'T'], aux_logits=True):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}
        self.aux_logits = aux_logits

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih , iw))  #改 // 2 
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih , iw))  #改 // 4 
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih , iw))   #改 // 8
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih-4, iw-4))   #改 // 16
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], ((ih - 6) , (iw - 6)))  #改 // 32 
            #block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih , iw))  #改 // 32 

        if self.aux_logits:                                                    #如果使用辅助分类器，即aux_logits = True，则创建aux1和aux2
            self.aux1 = InceptionAux(ih, num_classes,channels[2])                         #输入是Inception4a的输出
            self.aux2 = InceptionAux(ih-4, num_classes,channels[3])                         #输入是Inception4b的输出

        self.pool = nn.AvgPool2d(ih - 6, 1)  #改 // 32
        #self.pool = nn.AvgPool2d(ih , 1)  #改 // 32
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        
        x = self.s0(x)
        #print("s0:",x.shape) [50, 32, 9, 9]
        x = self.s1(x)
        #print("s1:",x.shape) [50, 32, 9, 9]
        x = self.s2(x)
        #print("s2:",x.shape) [50, 64, 9, 9]
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)
            

        x = self.s3(x)
        #print("s3:",x.shape) [50, 128, 5, 5]
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)
        x = self.s4(x)
        #print("s4:",x.shape) [50, 256, 3, 3]
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        #print("fc:",x.shape)
        
        if self.training and self.aux_logits:   # eval model lose this layer    是否使用辅助分类器，在训练过程使用，测试过程不用
            return x, aux2, aux1
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
            
            #layers.append(block(oup, oup, image_size, downsample=False))
        return nn.Sequential(*layers)


def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((9, 9), 100, num_blocks, channels, num_classes=160)


def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((9, 9), 100, num_blocks, channels, num_classes=160)


def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet((9, 9), 100, num_blocks, channels, num_classes=160)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((9, 9), 100, num_blocks, channels, num_classes=160)


def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((9, 9), 100, num_blocks, channels, num_classes=160)

def coatnet_6():
    num_blocks = [2, 2, 6, 14, 2]         # L
    channels = [32, 32, 64, 128, 256]   # D
    return CoAtNet((9, 9), 9, num_blocks, channels, num_classes=6)

def coatnet_7():
    num_blocks = [2, 2, 6, 14, 2]         # L
    channels = [32, 32, 64, 128, 256]   # D
    return CoAtNet((9, 9), 9, num_blocks, channels, num_classes=6,aux_logits=True)

def coatnet_8():
    num_blocks = [2, 2, 6, 14, 2]         # L
    channels = [32, 32, 64, 128, 256]   # D
    return CoAtNet((9, 9), 9, num_blocks, channels, num_classes=6,aux_logits=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 9, 9, 9)
    '''
    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_1()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_2()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_3()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_4()
    out = net(img)
    print(out.shape, count_parameters(net))
    '''

    net = coatnet_7()
    outputs , aux_logits2, aux_logits1 = net(img)
    print("outputs:",outputs.shape, count_parameters(net))
    print("aux_logits1:",aux_logits1.shape, count_parameters(net))
    print("aux_logits2:",aux_logits2.shape, count_parameters(net))
