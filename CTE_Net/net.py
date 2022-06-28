
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export

import math
from typing import Any

import torch
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module




from torch.nn import init
class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
import numpy as np
Gabor2 = np.array([[8.67955500e-17, 2.63136587e-12, 1.24794892e-09, 9.69570624e-09,
        1.24794892e-09, 2.63136587e-12, 8.67955500e-17],
       [1.91179921e-12, 5.79596904e-08, 2.74879043e-05, 2.13562142e-04,
        2.74879043e-05, 5.79596904e-08, 1.91179921e-12],
       [7.71274850e-10, 2.33826080e-05, 1.10894121e-02, 8.61571172e-02,
        1.10894121e-02, 2.33826080e-05, 7.71274850e-10],
       [5.69899314e-09, 1.72775402e-04, 8.19402877e-02, 6.36619772e-01,
        8.19402877e-02, 1.72775402e-04, 5.69899314e-09],
       [7.71274850e-10, 2.33826080e-05, 1.10894121e-02, 8.61571172e-02,
        1.10894121e-02, 2.33826080e-05, 7.71274850e-10],
       [1.91179921e-12, 5.79596904e-08, 2.74879043e-05, 2.13562142e-04,
        2.74879043e-05, 5.79596904e-08, 1.91179921e-12],
       [8.67955500e-17, 2.63136587e-12, 1.24794892e-09, 9.69570624e-09,
        1.24794892e-09, 2.63136587e-12, 8.67955500e-17]])

class TE_block_v3(nn.Module):
    def __init__(self,ch_in: int,kernel_size: int = 5,strides: int = 1,dropout=0.0,random_init = False, filters = 'Gabor2'):
        super().__init__()     
        # conv_texture
        self.conv_texture = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size,stride=strides,padding=int(kernel_size/2),bias=False)
        if not random_init:
            conv_filter = np.zeros((ch_in,ch_in,kernel_size,kernel_size))/(kernel_size**2)
            
            for i in range(ch_in):
                conv_filter[i,i,:,:] = Gabor2
            self.conv_texture.weight = torch.nn.Parameter(torch.tensor(conv_filter).type(torch.float32))
            print('Initialized TE block with Gabor filter')
        else:
            print('Initialized TE block with random')
        self.act = nn.PReLU()
    
        # SE block
        if ch_in>1:
            self.SEattention = SEAttention(channel = ch_in, reduction=8)
        else:
            self.SEattention = SEAttention(channel = ch_in, reduction=1)
            
        # dimension reduction
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_in, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_in),
            nn.PReLU(),)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t1: torch.Tensor = self.conv_texture(x)
        self.x1 = x_t1
        x_t2: torch.Tensor = self.act(x_t1) 
        self.x2 = x_t2
        x_t3: torch.Tensor = self.SEattention(x_t2)
        self.x3 = x_t3
        x_out1: torch.Tensor = torch.cat((x_t3,x),dim=1) 
        self.x4 = x_out1
        x_out2: torch.Tensor = self.conv_1x1(x_out1) 
        self.x5 = x_out2
        return x_out2

    
class CTE_Net(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
        random_init: bool = False,
        filters: str = 'Gabor2',
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            
            if len(channels) > 3:
                self.TE = TE_block_v3(inc, kernel_size = 7,random_init = random_init, filters = filters)
                return nn.Sequential(self.TE,self._get_connection_block(down, up, subblock))
            else:
                return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


