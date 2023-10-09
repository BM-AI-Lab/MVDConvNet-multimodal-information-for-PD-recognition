import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from OdConv import ODConv2d
from RDConv2d import RDConv2d

__all__ = ['Res2Net', 'res2net50']

model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}

def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)


def ddconv3x3_scale(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return RDConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


'''--------------------------------------------------------------------C and S-----------------------------------------------------------------'''
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
'''-----------------------------------------------------------------------------------------------------------------------------'''




class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        #self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(width * scale)
        self.conv1 = odconv1x1(inplanes, width*scale, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(width*scale)
        self.conv2 = odconv3x3(planes, width*scale, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(width*scale)
        self.c = 0

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(ddconv3x3_scale(width, width, stride, reduction=reduction, kernel_num=kernel_num))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = odconv1x1(width*scale, planes * self.expansion, reduction=reduction, kernel_num=kernel_num)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
        self.CBAM1 = CBAM(gate_channels=256)
        self.CBAM2 = CBAM(gate_channels=512)
        self.CBAM3 = CBAM(gate_channels=1024)
        self.CBAM4 = CBAM(gate_channels=2048)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        if out.shape[1] == 256:
            out = self.CBAM1(out)
        if out.shape[1] == 512:
            out = self.CBAM2(out)
        if out.shape[1] == 1024:
            out = self.CBAM3(out)
        if out.shape[1] == 2048:
            out = self.CBAM4(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        x = F.relu(out)

        return residual, out, x


class BasicStage(nn.Module):
    def __init__(self, block, inplanes, planes, blocks, stride=1, baseWidth=26, scale=4, stype='normal', r=16, L=64):
        super(BasicStage, self).__init__()

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=baseWidth, scale=scale))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, baseWidth=baseWidth, scale=scale))
        self.layers = nn.Sequential(*layers)

        d = max(int(planes * block.expansion / r), L)
        self.odconv = odconv1x1(inplanes, inplanes*1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(planes * block.expansion, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(blocks):
            self.fcs.append(
                nn.Conv2d(d, planes * block.expansion, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, block in enumerate(self.layers):
            residual, out, x = block(x)  # [b,c,w,h]
            fea_Zi = out.clone()
            if i == 0:
                fea_Z = fea_Zi.unsqueeze_(dim=1)
            else:
                fea_Z = torch.cat([fea_Z, fea_Zi.unsqueeze_(dim=1)], dim=1)

        fea_S = torch.sum(fea_Z, dim=1)
        fea_C = self.odconv(fea_S)
        fea_g = self.gap(fea_C)
        fea_v = self.fc(fea_g)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_v)
            if i == 0:
                vectors = vector.unsqueeze_(dim=1)
            else:
                vectors = torch.cat([vectors, vector.unsqueeze_(dim=1)], dim=1)
        vectors = self.softmax(vectors)
        fea_O = F.relu(torch.sum(fea_Z * vectors, dim=1))

        return fea_O








'''-----------------------------------------------------------------------------------------------------------------------------'''
class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=2):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #print("baseWidth", self.baseWidth)
        #print("scale", self.scale)
        self.stage1 = BasicStage(block, self.inplanes, 64, layers[0], baseWidth=self.baseWidth, scale=self.scale)
        self.inplanes = 256
        #self.CBAM1 = CBAM(gate_channels=256)
        self.stage2 = BasicStage(block, self.inplanes, 128, layers[1], stride=2, baseWidth=self.baseWidth,
                                 scale=self.scale)
        self.inplanes = 512
        #self.CBAM2 = CBAM(gate_channels=512)
        self.stage3 = BasicStage(block, self.inplanes, 256, layers[2], stride=2, baseWidth=self.baseWidth,
                                 scale=self.scale)
        self.inplanes = 1024
        #self.CBAM3 = CBAM(gate_channels=1024)
        self.stage4 = BasicStage(block, self.inplanes, 512, layers[3], stride=2, baseWidth=self.baseWidth,
                                 scale=self.scale)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        #x = self.CBAM1(x)
        x = self.stage2(x)
        #x = self.CBAM2(x)
        x = self.stage3(x)
        #x = self.CBAM3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = self.bn2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    model = Res2Net(Bottle2neck, [5, 6, 12, 5], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net101_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model


def res2net50_26w_6s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=6, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_6s']))
    return model


def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_8s']))
    return model


def res2net50_48w_2s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=48, scale=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_48w_2s']))
    return model


def res2net50_14w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_14w_8s']))
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net101_26w_4s(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())