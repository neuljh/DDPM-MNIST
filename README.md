# DDPM-MNIST
# Diffusion Model（DDPM）来生成MNIST数字 ， 如果有帮助希望你可以一键三连！！！

(1)初步完成
使用Diffusion Model(DDPM)模型来生成MNIST图片。这里生成的图片是数字0-9。

①　导入对应类库

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

②　定义关键的模型或类

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

此代码定义了一个残差卷积块，它是深度卷积神经网络的 ResNet 架构中使用的标准构建块。 ResidualConvBlock 是作为 nn.Module 的子类实现的，这是一个用于定义神经网络模块的 PyTorch 类。
ResidualConvBlock 具有三个输入参数：in_channels、out_channels 和 is_res。 in_channels 和 out_channels 是整数，分别表示卷积层的输入和输出通道数。 is_res 是一个布尔变量，表示该块是否为残差块。
init() 方法初始化块的参数，包括 self.same_channels 和 self.is_res，它们是实例变量，分别存储 in_channels==out_channels 和 is_res 参数的值。 然后，使用 nn.Sequential 类定义了两个卷积层，每个卷积层都包含一个二维卷积层 (nn.Conv2d)、一个批量归一化层 (nn.BatchNorm2d) 和一个 GELU 激活函数 (nn.GELU)。
GELU (Gaussian Error Linear Units) 是一种激活函数，其数学表达式为：

其中是标准正态分布的累积分布函数，erf 表示误差函数。GELU 激活函数被提出作为 ReLU (Rectified Linear Unit) 的一种改进，旨在通过克服 ReLU 的一些缺点来提高神经网络的性能。
与 ReLU 相比，GELU 在许多情况下表现更好。特别地，在自然语言处理和计算机视觉领域中，使用 GELU 激活函数的模型在一些基准测试中表现优于使用 ReLU 的模型。然而，GELU 的计算成本比 ReLU 更高，因为它涉及到计算误差函数和标准正态分布的累积分布函数，这可能会对训练速度和内存消耗产生一些影响。
forward() 方法执行 ResidualConvBlock 的正向传递。 如果 is_res 为 True，它首先将两个卷积层应用于输入张量 x，然后将输入张量 x 添加到输出张量。 这样做是为了在输入和输出张量之间创建一个“剩余”连接，允许梯度在反向传播期间直接流过块。 如果输入输出通道数不同（即self.same_channels为False），则输入张量x先经过第一个卷积层，第一个卷积层的输出与输出之间做残差连接 第二个卷积层。 最后，输出张量除以 2 的平方根（即 1.414）以考虑残差连接的缩放效应。
如果 is_res 为 False，则 forward() 方法只是将两个卷积层应用于输入张量 x 并返回输出张量。 这是针对 ResNet 架构中的非残差块完成的。

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)

此代码为 U-Net 架构定义了一个缩小块，这是一种用于图像分割任务的常用网络。 UnetDown 块作为 nn.Module 的子类实现，后者是用于定义神经网络模块的 PyTorch 类。
UnetDown 块有两个输入参数：in_channels 和 out_channels，它们是整数，分别表示块中卷积层的输入和输出通道数。
init() 方法初始化块的参数并定义块的层。 特别是，它创建了一个由 ResidualConvBlock 和 MaxPool2d 层组成的层列表（执行 2 倍的下采样），然后定义一个 nn.Sequential 模块，将这些层按顺序应用于输入张量。
ResidualConvBlock 是在代码其他地方定义的自定义模块，它由两个具有批量归一化和 GELU 激活的 2D 卷积层以及输入和输出张量之间的残差连接组成（如果 is_res 参数为 True）。 通过在 UnetDown 块中使用此 ResidualConvBlock，该块能够学习输入图像的更稳健和更具表现力的特征表示。
forward() 方法执行 UnetDown 块的正向传递。 它只是将 nn.Sequential 模块应用于输入张量并返回输出张量，由于 MaxPool2d 层，输出张量的空间维度是输入张量的一半。 这种下采样操作在 U-Net 架构中很重要，因为它允许网络捕获输入图像的高级和低级特征以用于分割任务。

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

此代码为 U-Net 架构定义了一个放大块，U-Net 架构是一种用于图像分割任务的常用网络。 UnetUp 块作为 nn.Module 的子类实现，后者是用于定义神经网络模块的 PyTorch 类。
UnetUp 块有两个输入参数：in_channels 和 out_channels，它们是整数，分别表示块中卷积层的输入和输出通道数。
init() 方法初始化块的参数并定义块的层。 特别是，它创建了一个层列表，其中包含一个 ConvTranspose2d 层（执行 2 倍的上采样），然后是两个 ResidualConvBlock 层。 ConvTranspose2d 层通过在相邻像素之间插入零然后将结果与可学习内核进行卷积来使输入张量的空间维度加倍。 ResidualConvBlock 层是在代码中其他地方定义的自定义模块，它们由两个具有批量归一化和 GELU 激活的 2D 卷积层组成，以及输入和输出张量之间的残差连接（如果 is_res 参数为 True）。 通过在 UnetUp 块中使用这些 ResidualConvBlock 层，该块能够学习输入图像的更稳健和更具表现力的特征表示。
forward() 方法执行 UnetUp 块的前向传递。 它需要两个输入张量：x，要上采样的张量，和 skip，要与 x 连接的张量。 该方法首先连接 x 并沿通道维度跳过，然后将 init() 方法中定义的 nn.Sequential 模块应用于连接后的张量。 最后，它返回输出张量，由于 ConvTranspose2d 层，它的空间维度是输入张量的两倍。

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

此代码定义了一个名为 EmbedFC 的神经网络模块，它使用单个完全连接层 (FC) 执行输入数据的嵌入。 EmbedFC 模块作为 nn.Module 的子类实现，后者是用于定义神经网络模块的 PyTorch 类。
EmbedFC 模块有两个输入参数：input_dim 和 emb_dim，它们分别是表示输入维度和嵌入空间维度的整数。
init() 方法初始化 EmbedFC 模块的参数并定义模块的层。 特别是，它创建了一个层列表，其中包含一个 FC 层，后跟一个 GELU 激活函数，以及另一个输出最终嵌入的 FC 层。 FC 层采用形状为 (batch_size, input_dim) 的输入张量并对它应用线性变换，然后是 GELU 激活函数。 GELU 激活函数是整流线性单元 (ReLU) 函数的一个变体，已被证明可以提高某些任务的性能。 第二个 FC 层将第一个 FC 层的输出映射到具有维度 emb_dim 的嵌入空间。
forward() 方法执行 EmbedFC 模块的前向传递。 它采用输入张量 x 并首先将其重塑为形状为 (-1, input_dim) 的二维张量，其中 -1 表示推断的批量大小。 然后，重塑后的张量通过 init() 方法中定义的 nn.Sequential 模块传递，该模块按顺序应用两个 FC 层和 GELU 激活函数。 最后，返回具有形状 (batch_size, emb_dim) 的输出张量。

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)
        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)  
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask      
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)
        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

此代码定义了一个名为 ContextUnet 的 PyTorch 模块，它是用于图像分割任务的 U-Net 架构的修改版本。 修改后的架构将额外的上下文和时间信息合并到模型中。
__init__ 函数定义了模型的架构。 输入参数是 in_channels（图像的输入通道数）、n_feat（隐藏层中的特征/通道数）和 n_classes（分割任务中的类数）。
init_conv 层是一个残差卷积块（定义在一个单独的模块 ResidualConvBlock 中），它将输入图像作为输入并输出具有 n_feat 通道的特征图。
down1 和 down2 层是两个“下采样”块，它们通过应用卷积层和最大池化顺序减小特征图的空间大小。 down1以init_conv的输出为输入，输出一个n_feat通道的feature map，而down2以down1的输出为输入，输出一个2*n_feat通道的feature map。
to_vec 层是一个平均池化层，后跟一个 GELU 激活函数。 它将down2的输出作为输入，输出一个维度为2*n_feat的向量。
timeembed1、timeembed2、contextembed1 和 contextembed2 层是嵌入层，它们将时间和上下文输入转换为可以与图像特征图连接的特征向量。 EmbedFC 模块定义了一个具有 GELU 激活函数的全连接神经网络。
up0、up1 和 up2 层是“上采样”块，它们通过应用卷积转置层，然后是组归一化和 ReLU 激活函数，依次增加特征图的空间大小。 up0 将 to_vec 的输出作为输入，输出一个具有 2*n_feat 通道的特征图。 up1 将 down2 和 up0 的连接特征图作为输入，并输出具有 n_feat 通道的特征图。 up2 将 down1 和 up1 的连接特征图作为输入，并输出具有 n_feat 通道的特征图。
最后，out 层是一个卷积层，它将来自 up2 的级联特征图和原始输入图像作为输入，并输出具有 in_channels 通道的特征图。
forward() 方法定义了神经网络模型的前向传递，该模型将图像、上下文标签、时间步长和上下文掩码作为输入，并生成输出图像作为输出。 该模型使用 U-Net 架构，通常用于图像分割任务。
前向函数有四个输入：x，表示输入图像的张量； c，表示上下文标签的整数； t，表示时间步长的整数； 和 context_mask，一个二进制张量，指示哪些样本应该屏蔽掉上下文信息。
输入图像 x 首先通过初始卷积层 (init_conv)，然后通过两个下采样层（down1 和 down2）以生成特征图。 最终的特征图down2然后通过一个全连接层（to_vec）产生一个隐藏向量hiddenvec。
上下文标签 c 首先使用 one_hot 函数转换为 one-hot 编码。 然后将生成的单热张量乘以掩码 context_mask 以屏蔽某些样本的上下文信息。 通过沿第二个维度将 context_mask 张量 self.n_classes 复制多次，然后将掩码的值从 0 翻转到 -1 以及从 1 翻转到 0 来创建掩码。
然后使用四个不同的嵌入层将单热张量 c 和时间步长 t 嵌入到四个独立的特征图中：contextembed1、timeembed1、contextembed2 和 timeembed2。 生成的特征图被重塑为大小为 (batch_size, n_feat*2, 1, 1) 或 (batch_size, n_feat, 1, 1)，其中 n_feat 是表示隐藏向量中特征数量的超参数。
最后，上采样层使用了两次。首先，up1是通过将hiddenvec传递给self.up0来计算的。接下来，up2和up3都使用了上一层的输出作为它们的输入，同时也使用了上下文和时间嵌入。

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
这段代码返回预先计算好的用于 DDPM（Diffusion Probabilistic Models）采样和训练过程中需要用到的各种变量和系数。
beta1 和 beta2 是两个参数，要求满足 beta1 < beta2 < 1.0，否则会抛出 AssertionError 异常。这两个参数在 DDPM 的采样和训练过程中起到很重要的作用，它们决定了每个时间步的噪声水平，随着时间步的增加，噪声水平逐渐变大。其中，beta1 表示起始噪声水平，beta2 表示结束噪声水平。
T 表示总的时间步数。
beta_t 表示每个时间步的噪声水平，其取值从 beta1 逐渐增加到 beta2。
sqrt_beta_t 表示 beta_t 的平方根。
alpha_t 表示每个时间步的衰减系数，其取值从 1 减少到 beta_t。
log_alpha_t 表示 alpha_t 的自然对数。
alphabar_t 表示每个时间步的衰减系数的累积乘积。
sqrtab 表示 alphabar_t 的平方根。
oneover_sqrta 表示 alpha_t 的倒数的平方根。
sqrtmab 表示 1 - alphabar_t 的平方根。
mab_over_sqrtmab 表示 (1 - alpha_t) / sqrtmab，也就是 sqrtmab 的倒数乘以 1 - alpha_t。
这些变量和系数在 DDPM 的采样和训练过程中用到，比如在采样过程中，需要根据这些变量和系数计算出每个时间步的噪声水平和衰减系数，以及计算出噪声的标准差；在训练过程中，需要使用这些变量和系数计算出损失函数。

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)  
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)
        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free
        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)
            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

此代码定义了一个名为 DDPM（代表扩散概率模型）的 PyTorch 模块，用于从扩散模型进行训练和采样。 类构造函数使用 nn_model 参数初始化模型，这是一个用于扩散模型的神经网络模型。 betas 参数是一个元组，包含两个确定模型学习率的浮点数。 n_T 参数是一个整数，它确定在训练或采样过程中采用的扩散步骤数。 设备参数指定运行模型的设备（例如 CPU 或 GPU）。 drop_prob 参数指定在训练期间应用 dropout 的概率。 loss_mse 属性是均方误差损失函数的一个实例。
Forward()方法用于训练模型。 它接受两个张量 x 和 c。 x 表示输入数据，c 是上下文向量。 该方法为批次中的每个样本生成一个介于 1 和 n_T 之间的随机整数 _ts。 然后它生成一个与 x 形状相同的随机张量噪声。 使用 _ts 和噪声，该方法生成一个张量 x_t，它表示步骤 _ts 的扩散过程的状态。 然后使用神经网络模型从 x_t 和 c 预测“误差项”，并返回预测误差与原始噪声张量之间的均方误差损失。
Sample()方法用于从扩散模型中采样。 它接受 n_sample，一个表示要生成的样本数量的整数，size，一个表示生成样本大小的元组，device，指定模型将运行的设备，以及一个可选参数 guide_w，它确定指导级别 在采样期间。 该方法生成具有给定大小的 n_sample 随机张量 x_i 和循环数字 0 到 9 的上下文张量 c_i。该方法然后用零初始化张量 context_mask，表示正在使用的上下文，并通过重复将批量大小加倍 c_i 和 context_mask。 一半的批次将 context_mask 设置为 1，表示未使用上下文。 该方法然后通过从 n_T 向后迭代到 1 生成 n_T 个样本。在每一步，该方法生成一个随机张量 z，将神经网络模型的预测分为两部分，eps1 和 eps2，并通过混合生成一个新的张量 eps eps1 和 eps2 与 guide_w。 然后，该方法通过将 x_i 与 eps、z 和模型初始化期间存储的几个张量组合来生成新的张量 x_i。 生成的样本以两种形式返回，张量 x_i 和包含扩散过程中间步骤的 numpy 数组 x_i_store。

③　定义训练过程
定义一些超参数：
n_epoch = 20
batch_size = 256
n_T = 400 # 500
device = "cuda:0"
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
lrate = 1e-4
save_model = False
save_dir = './data/diffusion_outputs10/'
ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
此代码初始化几个超参数，这些超参数将在稍后的训练和测试过程中使用：
n_epoch 是用于训练模型的纪元数（即，完全通过训练数据）。
batch_size 是将在训练期间一次处理的数据点（即本例中的图像）的数量。 较大的批量通常会导致更快的训练，但也需要更多的内存。
n_T 是从模型生成样本时要采取的扩散步骤数。 这决定了输入到模型的随机噪声序列的长度，并影响生成样本的质量。
device 是运行模型的设备（例如 CPU 或 GPU）。
n_classes 是数据集中的类数。 在这种情况下，它是 10，因为数据集是 MNIST，它有 10 个类别对应于数字 0-9。
n_feat 是模型卷积层中特征（即通道）的数量。 较大的值通常会产生更具表现力的模型，但需要更多的内存并且可能会过度拟合训练数据。
lrate 是优化器在训练期间使用的学习率。 这决定了在损失函数的负梯度方向上采取的步长，并影响模型收敛到良好解决方案的速度。
save_model 是一个标志，指示训练后是否将训练好的模型保存到磁盘。
save_dir 是将训练模型保存到的目录（如果 save_model 为 True）。
ws_test 是三个值的列表，代表测试期间生成指导的强度。 值 0.0 对应于无指导，0.5 对应于适度指导，2.0 对应于强指导。 这样做的目的是查看模型在生成具有不同指导级别的样本时的表现如何。

ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
ddpm.to(device)
tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 
dataset = MNIST("./data", train=True, download=True, transform=tf)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

该代码为 MNIST 数据集初始化并建立了称为 DDPM（去噪扩散概率模型）的深度生成模型。
ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1) 初始化DDPM模型。 它采用以下参数：
nn_model：扩散模型使用的神经网络模型。 在这种情况下，ContextUnet 与 1 个输入通道、n_feat 隐藏特征和 n_classes 输出类一起使用。
betas：Adam 优化器中使用的 beta。
n_T：运行模型的扩散步数或时间步数。
device：运行模型的设备。
drop_prob：在训练期间应用 dropout 的概率。
ddpm.to(device) 将模型移动到指定的设备。
dataset = MNIST("./data", train=True, download=True, transform=tf) 从指定目录下载并创建 MNIST 数据集实例，应用 tf 中定义的转换，并将其设置为训练。 train=True 表示数据集用于训练。
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5) 创建一个数据加载器，生成用于训练的数据批次。 它采用以下参数：
Dataset：用于生成批次的数据集。
batch_size：每批样本的数量。
shuffle：是否在 epoch 之间打乱数据。
num_workers：用于数据加载的子进程数。
optim = torch.optim.Adam(ddpm.parameters(), lr=lrate) 创建一个 Adam 优化器以在训练期间更新 DDPM 模型的参数。 ddpm.parameters() 指定要优化的参数，lr 指定优化器的学习率。


for ep in range(n_epoch):
    print(f'epoch {ep}')
    ddpm.train()
    # linear lrate decay
    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
    pbar = tqdm(dataloader)
    loss_ema = None
    for x, c in pbar:
        optim.zero_grad()
        x = x.to(device)
        c = c.to(device)
        loss = ddpm(x, c)
        loss.backward()
        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        optim.step()
    # for eval, save an image of currently generated samples (top rows)
    # followed by real images (bottom rows)
    ddpm.eval()
    with torch.no_grad():
        n_sample = 4*n_classes
        for w_i, w in enumerate(ws_test):
            x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)
            # append some real images at bottom, order by class also
            x_real = torch.Tensor(x_gen.shape).to(device)
            for k in range(n_classes):
                for j in range(int(n_sample/n_classes)):
                    try: 
                        idx = torch.squeeze((c == k).nonzero())[j]
                    except:
                        idx = 0
                    x_real[k+(j*n_classes)] = x[idx]

            x_all = torch.cat([x_gen, x_real])
            grid = make_grid(x_all*-1 + 1, nrow=10)
            print("grid: ")
            print(grid)
            save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
            print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

            if ep%5==0 or ep == int(n_epoch-1):
                # create gif of images evolving over time, based on x_gen_store
                fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                def animate_diff(i, x_gen_store):
                    print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                    plots = []
                    for row in range(int(n_sample/n_classes)):
                        for col in range(n_classes):
                            axs[row, col].clear()
                            axs[row, col].set_xticks([])
                            axs[row, col].set_yticks([])
                            # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                            plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                    return plots
                ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
    # optionally save model
    if save_model and ep == int(n_epoch-1):
        torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
        print('saved model at ' + save_dir + f"model_{ep}.pth")

训练循环开始，迭代指定的时期数。 DDPM 设置为训练模式，学习率随每个 epoch 线性衰减。 tqdm() 方法用于在训练期间显示进度条。
loss_ema 变量用于计算损失的指数移动平均值以用于显示目的。 输入图像和标签被加载到设备。 通过使用输入图像和标签调用 DDPM 实例来计算损失，并执行反向传播。 loss_ema 值已更新，进度条已更新为新值。
在每个 epoch 之后，DDPM 被设置为评估模式，并为 ws_test 列表中的每个 w 值生成并保存图像网格。 x_gen 变量是通过调用 DDPM 实例的 sample() 方法生成的，该实例从模型中生成样本。 x_real 变量是一个包含来自数据集的真实图像的张量。 x_all 变量将生成的图像和真实图像连接成一个张量。 图像网格是使用 torchvision.utils 中的 make_grid() 方法创建的，并使用 save_image() 方法保存。 如果当前 epoch 是 5 的倍数或最后一个 epoch，则使用当前 epoch 生成的图像生成 gif 并保存。
在训练循环结束时，如果指定，则保存训练的模型。

④　实验结果
在前文定义了变量ws_test ，它是三个值的列表，代表测试期间生成指导的强度。 值 0.0 对应于无指导，0.5 对应于适度指导，2.0 对应于强指导。 这样做的目的是查看模型在生成具有不同指导级别的样本时的表现如何。因此对于每个epoch，存在无指导、适度指导和强指导的训练情况。

A.实验结果(epochs=20,learning_rate=0.0001,batch_size=256)：
下面将展示epochs从0到19的动态变化图：
1)实验结果(epochs=0,ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/78b0f0db-dc6a-4c41-b366-ca9b2b3f9320)

2)实验结果(epochs=1;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/8ed32f80-5e5b-461f-a92d-7b2ee0980a75)

3)实验结果(epochs=2;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/c5f1588e-a8a8-4711-b0a7-534e8c2644bf)

4)实验结果(epochs=3;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/a2e0a23f-1b8c-4675-8861-947d45e7e067)

5)实验结果(epochs=4;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/e9b2ecfe-59c0-4a40-bd8f-bd161bae2330)

6)实验结果(epochs=5;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/b05d418b-8f63-44a0-a7cf-54137a2a18e6)

7)实验结果(epochs=6;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/4496446c-0c04-416a-9493-230f810d68b9)

8)实验结果(epochs=7;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/9b2e15ee-6c7a-4c9a-881a-17b5e6749f50)

9)实验结果(epochs=8;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/f75168f6-8fd4-4150-9ff8-c16e897f85f1)

10)实验结果(epochs=9;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/b5e32adc-f1fa-4bae-9d51-aacb3aebcb5b)

11)实验结果(epochs=10;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/2e686bda-0d06-4439-b2f9-50779ad0e1bd)

12)实验结果(epochs=11;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/c472d32d-06b1-471c-9427-4c49cedf6acf)

13)实验结果(epochs=12;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/05087d3a-d099-4b34-bf71-db365e63905c)

14)实验结果(epochs=13;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/5fb14eef-1ffa-4cf1-9335-871e2d5a305e)

15)实验结果(epochs=14;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/a58328ae-8520-4c90-bda8-0a15281f3529)

16)实验结果(epochs=15;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/7a0bf6ff-0872-47ae-afc5-7853903a4b45)

17)实验结果(epochs=16;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/649c8d45-bc70-40f1-b1f4-a16af87e8068)

18)实验结果(epochs=17;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/499ed1b5-ff3c-4db1-a551-92c8c744bd15)

19)实验结果(epochs=18;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/11d896a1-51af-4132-b89b-c5f6b06dac44)

20)实验结果(epochs=19;ws_test=0.0,0.5,2.0)：
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/3bf6b592-cb06-4b42-b017-4979300104a6)


(2)与普通GAN的对比
为了保证对比的准确性，我们调整GAN模型和DDPM模型的参数基本一致：
Epochs=20
Batch_size=256
Learning_rate=0.0001
1.epoch=10
普通GAN(epoch=10):
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/db29ca7f-a87e-4863-a5a3-38292b8744ea)

DDPM(epoch=10;ws_test=0.0,0.5,2.0):
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/5ae277c1-9c9f-47d5-8adc-6c91694f249c)


2.epoch=20
普通GAN(epoch=20):
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/b3c82f17-b413-4034-b42a-c96b5de2bde4)

DDPM(epoch=20;ws_test=0.0,0.5,2.0):
![image](https://github.com/neuljh/DDPM-MNIST/assets/132900799/89016338-0ac3-4273-b7c0-16f99de7d792)


3.对比总结
通过对比我们可以很直观的看出，DDPM在性能方面明显优于普通GAN，但是明显的，在训练模型方面，DDPM需要的时间和资源也远远超过于普通GAN。
