import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.jit.annotations import Optional, Tuple

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 #if downsample == False else 2 改
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        # nn.ReLU(inplace=True)

    )
    return layer

def swish(x):
    return x*torch.sigmoid(x)   

class residual_block(nn.Module):
    

    def __init__(self, in_channel, out_channel, image_size, downsample=False):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.drop2 = nn.Dropout(0.5)
        self.conv3 = conv3x3x3(out_channel, out_channel) 
        self.drop3 = nn.Dropout(0.5)

    def forward(self, x):  # (1,1,100,9,9)
        #x1 = F.relu(self.conv1(x), inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        #x2 = F.relu(self.conv2(x1), inplace=True)  # (1,8,100,9,9) (1,16,25,5,5)
        #x3 = self.conv3(x2)  # (1,8,100,9,9) (1,16,25,5,5)
        #out = F.relu(x1 + x3, inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        x1=swish(self.conv1(x))
        x1=self.drop1(x1)
        x2=swish(self.conv2(x1))   
        x2=self.drop1(x2)                    
        x3=swish(self.conv3(x2))
        x3=self.drop1(x3)
        out=swish(x1+x3)                    
        
        return out
#############################################################################################################
class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
      
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
    
class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.5):
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
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.5):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        
        self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
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
            x0 = self.proj(x) #[50, 128, 5, 5]
            x = x0 + self.attn(x)  #[50, 128, 5, 5]
        x = x + self.ff(x)
        return x
    
#############################################################################################################
class APSMnet_Drop(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=6, block_types=['C', 'T', 'C', 'T'], aux_logits=True):
        super().__init__()
        ih, iw = image_size
        block = {'C': residual_block, 'T': Transformer}
        self.aux_logits = aux_logits

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih , iw))  #改 // 2 
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih , iw))  #改 // 4 
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2) 
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih-4 , iw-4))   #改 // 8
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih-4 , iw-4))   #改 // 16
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2) 
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], ((ih - 6) , (iw - 6)))  #改 // 32 
       
        self.spt1 = self._make_layer(
            block[block_types[0]], channels[0], channels[4], num_blocks[1], (ih , iw))  #改 // 4 
        self.spt2 = self._make_layer(
            block[block_types[1]], channels[0], channels[4], num_blocks[2], (ih , iw))   #改 // 8
        self.spt3 = self._make_layer(
            block[block_types[2]], channels[0], channels[4], num_blocks[3], (ih, iw))   #改 // 16
        self.spt4 = self._make_layer(
            block[block_types[3]], channels[0], channels[4], num_blocks[4], (ih , iw))  #改 // 32 
            #block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih , iw))  #改 // 32 

        if self.aux_logits:                                                    #如果使用辅助分类器，即aux_logits = True，则创建aux1和aux2
            self.aux1 = InceptionAux(ih-6, num_classes,channels[4])                         #输入是Inception4a的输出
            self.aux2 = InceptionAux(ih, num_classes,1024)                         #输入是Inception4b的输出
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, padding=0, stride=2) 
        self.proj = nn.Conv2d(1024, channels[4], 1, 1, 0, bias=False)
        self.pool = nn.AvgPool2d(ih - 6, 1)  #改 // 32
        #self.pool = nn.AvgPool2d(ih , 1)  #改 // 32
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        
        x0 = self.s0(x) #[50, 32, 9, 9]       
        x1 = self.s1(x0) #[50, 64, 9, 9]      
        x1_m = self.maxpool1(x1) #[50, 64, 5, 5]
        x2 = self.s2(x1_m) #[50, 64, 5, 5]
        x3 = self.s3(x2) #[50, 256, 5, 5]
        x3_m = self.maxpool2(x3) #[50, 256, 3, 3]
        x4 = self.s4(x3_m) #[50, 256, 3, 3]
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x4)
            
        
        sp1 = self.spt1(x0) #[50, 256, 9, 9] C
        sp2 = self.spt2(x0) #[50, 256, 9, 9] T
        sp3 = self.spt3(x0) #[50, 256, 9, 9] C
        sp4 = self.spt4(x0) #[50, 256, 9, 9] T
        add1 = sp1 + sp2  #[50, 256, 9, 9]
        add2 = add1 + sp3  #[50, 256, 9, 9]
        add3 = add2 + sp4  #[50, 256, 9, 9]
        combine = torch.cat([sp1, add1, add2, add3], 1) #[50, 1024, 9, 9]

        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(combine) #[50, 6]
        
        combine = self.maxpool3(combine) #[50, 1024, 3, 3]
        combine = self.proj(combine) #[50, 256, 3, 3]
       
        x_all = x4 + combine #[50, 256, 3, 3]
        x_all = self.pool(x_all).view(-1, x_all.shape[1]) #[50, 256]
        x_all = self.fc(x_all) #[50, 6]
 

        if self.training and self.aux_logits:   # eval model lose this layer    是否使用辅助分类器，在训练过程使用，测试过程不用
            return x_all, aux2, aux1
        return x_all

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=False))
            else:
                layers.append(block(oup, oup, image_size))
            
            #layers.append(block(oup, oup, image_size, downsample=False))
        return nn.Sequential(*layers)
    
#############################################################################################################



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




def APSMnet_Drop1():
    num_blocks = [1, 1, 1, 1, 1]         # L
    channels = [32, 64, 128, 256, 256]   # D
    return APSMnet_Drop((9, 9), 9, num_blocks, channels, num_classes=6,aux_logits=True)


def APSMnet_Drop2():
    num_blocks = [1, 1, 1, 1, 1]         # L
    channels = [32, 64, 128, 256, 256]   # D
    return APSMnet_Drop((9, 9), 9, num_blocks, channels, num_classes=6,aux_logits=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(50, 9, 9, 9)

    net = APSMnet_Drop1()
    outputs , aux_logits2, aux_logits1 = net(img)
    print("outputs:",outputs.shape, count_parameters(net))
    print("aux_logits1:",aux_logits1.shape, count_parameters(net))
    print("aux_logits2:",aux_logits2.shape, count_parameters(net))

    