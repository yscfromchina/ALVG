import torch
from torch import nn
import torch.nn.functional as F
from detectron2.layers import Conv2d
from .img_enhance import ImgEnhance
# from .deform_conv import DFConv2d
from detectron2.layers.batch_norm import get_norm


def conv_with_kaiming_uniform(
        norm=None, activation=None,
        use_deformable=False, use_sep=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        if use_deformable:
            # conv_func = DFConv2d
            assert("deformable is not supported for now")
        else:
            conv_func = Conv2d
        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1
        conv = conv_func(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=groups,
            bias=(norm is None)
        )
        if not use_deformable:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(conv.weight, a=1)
            if norm is None:
                nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if norm is not None and len(norm) > 0:
            if norm == "GN":
                norm_module = nn.GroupNorm(32, out_channels)
            else:
                norm_module = get_norm(norm, out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


class Easy_Mask_Head(nn.Module):
    def __init__(self, in_channels, channels=128, num_heads=8, dropout=0.1, activation='ReLU'):
        super(Easy_Mask_Head, self).__init__()
        conv_block = conv_with_kaiming_uniform("BN", activation=True)

        self.cross_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(in_channels)
        self.refine = conv_block(in_channels, channels, 3)
        if activation == 'ReLU':
            activation = nn.ReLU(inplace=True)
        else:
            activation = nn.LeakyReLU(inplace=True)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(num_features=channels // 2),
            activation,
            nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=4, stride=2, padding=1),   # Upsample
            nn.BatchNorm2d(num_features=channels // 4),
            activation,
            nn.ConvTranspose2d(channels // 4, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, features, pos_embed):

        bs, c, h, w = features.shape
        # num_queries = hidden_states.shape[1]
        # pos_embed = pos_embed.view(bs, c, -1)
        # features = features.view(bs, c, -1)
        features = self.with_pos_embed(pos_embed, features)

        # _features = F.normalize(features, p=2, dim=-1)
        # hidden_states = F.normalize(hidden_states, p=2, dim=-1)
        # score = torch.bmm(hidden_states, _features)
        # score = 2 * torch.sigmoid(score)
        # features = features.transpose(1, 2).unsqueeze(1).repeat(1, num_queries, 1, 1)
        # features = score.unsqueeze(3) * features
        # features = features.view(-1, h, w, c).permute(0, 3, 1, 2)

        x = self.refine(features)
        mask_feats = self.upsample(x)
        mask_feats = mask_feats.view(bs, h * 8, w * 8)
        return mask_feats
