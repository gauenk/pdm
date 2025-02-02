from abc import abstractmethod
import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import stnls

from .nn import (
    checkpoint,
    zero_module,
    normalization,
    count_flops_attn,
    gamma_embedding
)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """

class FlowsBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, flows):
        """
        Apply the module to `x` given `emb` embeddings.
        """



class MultiSequential(nn.Sequential, FlowsBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, flows):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                B = x.shape[0]
                x = rearrange(x,'b t c h w -> (b t) c h w')
                x = layer(x, emb)
                x = rearrange(x,'(b t) c h w -> b t c h w',b=B)
            elif isinstance(layer, FlowsBlock):
                x = layer(x, flows)
            else:
                B = x.shape[0]
                x = rearrange(x,'b t c h w -> (b t) c h w')

                x = layer(x)

                x = rearrange(x,'(b t) c h w -> b t c h w',b=B)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channel, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channel, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channel
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of embedding channels.
    :param dropout: the rate of dropout.
    :param out_channel: if specified, the number of out channels.
    :param use_conv: if True and out_channel is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channel=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            nn.Conv2d(channels, self.out_channel, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channel),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
            ),
        )

        if self.out_channel == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channel, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # B = x.shape[0]
        # print("vid.shape: ",x.shape)
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        # print("[emb]: ",emb.shape)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            # print("emb_out.shape: ",emb_out.shape)
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            # scale, shift = torch.chunk(emb_out, 2, dim=1)
            # print(h.shape,out_norm)
            # print(scale.shape,shift.shape)
            # print("[shape]: ",h.shape,scale.shape,shift.shape)
            B = scale.shape[0]
            h = rearrange(h,'(b t) c h w -> b t c h w',b=B)
            hbs = []
            for b in range(B):
                hb = out_norm(h[b]) * (1 + scale[[b]]) + shift[[b]]
                hb = out_rest(hb)
                hbs.append(hb)
            hbs = th.stack(hbs)
            h = rearrange(hbs,'b t c h w -> (b t) c h w',b=B)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
            attn_cfg=None,
            search_cfg=None,
            normz_cfg=None,
            agg_cfg=None

    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class NonLocalAttentionWrap(FlowsBlock):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        attn_cfg=None,
        search_cfg=None,
        normz_cfg=None,
        agg_cfg=None
    ):
        super().__init__()

        # -- previous template --
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        # -- wrapped non-local attention module --
        attn_cfg.embed_dim = channels
        self.attn = stnls.nn.NonLocalAttention(attn_cfg,search_cfg,normz_cfg,agg_cfg)

    def forward(self, vid, flows=None):
        # print("vid.shape: ",vid.shape)
        # vid = vid[None,:]
        vid = self.attn(vid, flows)
        # print("vid.shape: ",vid.shape)
        return vid


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class UNet(nn.Module):
    """
    The full UNet model with attention and embedding.
    :param in_channel: channels in the input Tensor, for image colorization : Y_channels + X_channels .
    :param inner_channel: base channel count for the model.
    :param out_channel: channels in the output Tensor.
    :param res_blocks: number of residual blocks per downsample.
    :param attn_res: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mults: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        # in_channel,
        # inner_channel,
        # out_channel,
        # res_blocks,
        # attn_res,
        # dropout=0,
        # channel_mults=(1, 2, 4, 8),
        # conv_resample=True,
        # use_checkpoint=False,
        # use_fp16=False,
        # num_heads=1,
        # num_head_channels=-1,
        # num_heads_upsample=-1,
        # use_scale_shift_norm=True,
        # resblock_updown=True,
        # use_new_attention_order=False,
        arch_cfg,
        attn_cfg=None,
        search_cfg=None,
        normz_cfg=None,
        agg_cfg=None
    ):

        super().__init__()

        # -- unpack --
        use_checkpoint = arch_cfg.use_checkpoint
        attn_res = arch_cfg.attn_res
        resblock_updown = arch_cfg.resblock_updown
        num_heads=arch_cfg.num_heads
        num_head_channels=arch_cfg.num_head_channels
        num_heads_upsample=arch_cfg.num_heads_upsample
        use_new_attention_order=arch_cfg.use_new_attention_order
        in_channel = arch_cfg.in_channel
        out_channel = arch_cfg.out_channel
        inner_channel = arch_cfg.inner_channel
        channel_mults = arch_cfg.channel_mults
        res_blocks = arch_cfg.res_blocks
        dropout = arch_cfg.dropout
        use_checkpoint = arch_cfg.use_checkpoint
        use_new_attention_order=arch_cfg.use_new_attention_order
        out_ch = arch_cfg.out_channel
        conv_resample = arch_cfg.conv_resample
        attn_type = arch_cfg.attn_type
        use_scale_shift_norm = arch_cfg.use_scale_shift_norm
        cond_embed_dim = inner_channel * 4
        # print("inner_channel: ",inner_channel)

        # -- init --
        self.inner_channel = inner_channel
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if attn_type == "stnls":
            AttBlk = NonLocalAttentionWrap
        else:
            AttBlk = AttentionBlock
        cond_embed_dim = inner_channel * 4
        self.cond_embed = nn.Sequential(
            nn.Linear(inner_channel, cond_embed_dim),
            SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        ch = input_ch = int(channel_mults[0] * inner_channel)
        self.input_blocks = nn.ModuleList(
            [MultiSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(mult * inner_channel),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * inner_channel)
                if ds in attn_res:
                    layers.append(
                        AttBlk(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            attn_cfg=attn_cfg,
                            search_cfg=search_cfg,
                            normz_cfg=normz_cfg,
                            agg_cfg=agg_cfg,
                        )
                    )
                self.input_blocks.append(MultiSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    MultiSequential(
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, out_channel=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = MultiSequential(
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttBlk(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                attn_cfg=attn_cfg,
                search_cfg=search_cfg,
                normz_cfg=normz_cfg,
                agg_cfg=agg_cfg,
            ),
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(inner_channel * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(inner_channel * mult)
                if ds in attn_res:
                    layers.append(
                        AttBlk(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            attn_cfg=attn_cfg,
                            search_cfg=search_cfg,
                            normz_cfg=normz_cfg,
                            agg_cfg=agg_cfg,
                        )
                    )
                if level and i == res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channel=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(MultiSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
        )

    def forward(self, vid, gammas, flows):
        """
        Apply the model to an input batch.
        :param x: an [B x T x C0 x ...] Tensor of inputs (B&W)
        :param gammas: a Tx? 2-D batch of gammas.
        :param flows: a 5-D batch of optical flow tensors
        :return: an [B x T x C1 x ...] Tensor of outputs.
        """

        B,NF,*_ = vid.shape
        hs = []
        gammas = gammas.view(B, )
        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))
        # emb = emb.view(B,NF,-1)
        # print("\n\n[vid,gammas]: ",vid.shape,gammas.shape)

        h = vid.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb, flows)
            # print("[h,emb]: ",h.shape,emb.shape)
            hs.append(h)
        h = self.middle_block(h, emb, flows)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=2)
            h = module(h, emb, flows)
            # print("[h,emb]: ",h.shape,emb.shape)
        h = h.type(vid.dtype)

        # -- output --
        B = h.shape[0]
        h = rearrange(h,'b t c h w -> (b t) c h w')
        # print("[h0]: ",h.shape)
        h = self.out(h)
        h = rearrange(h,'(b t) c h w -> b t c h w',b=B)
        # print("[h1]: ",h.shape)

        return h

if __name__ == '__main__':
    b, c, h, w = 3, 6, 64, 64
    timsteps = 100
    model = UNet(
        in_channel=c,
        inner_channel=64,
        out_channel=3,
        res_blocks=2,
        attn_res=[8]
    )
    x = torch.randn((b, c, h, w))
    emb = torch.ones((b, ))
    out = model(x, emb)
