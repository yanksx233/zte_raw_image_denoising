import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): Window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    window_size, C = windows.shape[-2:]
    windows = windows.view(-1, H // window_size, W // window_size, window_size, window_size, C)
    x = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def to_image(x, x_size):
    """
    Args:
        x (tensor): shape of (B, L, C)
        x_size (tuple[int]): (H, W)
        returns (tensor): (B, C, H, W)
    """
    B, L, C = x.shape
    H, W = x_size
    x = x.transpose(1, 2).contiguous().view(B, C, H, W)
    return x


def to_token(x):
    """
    Args:
        x (tensor): shape of (B, C, H, W)
        returns (tensor): (B, L, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H * W).transpose(1, 2).contiguous()
    return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        num_feats (int): Number of input channels.
        window_size (int): window_size
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, num_feats, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.window_size = window_size  # WS
        self.num_heads = num_heads
        self.scale = qk_scale or (num_feats // num_heads) ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, WS, WS)
        coords_flatten = torch.flatten(coords, 1)  # (2, WS*WS)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, WS*WS, WS*WS)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (WS*WS, WS*WS, 2)
        relative_coords += window_size - 1  # shift to start from 0
        relative_coords[:, :, 0] *= 2 * window_size - 1  # h range different from w
        relative_position_index = relative_coords.sum(-1)  # (WS*WS, WS*WS)  1D displace
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(num_feats, num_feats * 3, bias=qkv_bias)
        self.att_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(num_feats, num_feats)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, WS*WS, C)
            mask: (0/-inf) mask with shape of (num_windows, WS*WS, WS*WS) or None
        """
        B_, N, C = x.shape  # N == WS*WS
        qkv = self.qkv(x).view(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, WS*WS, C_)

        att = (q * self.scale) @ k.transpose(2, 3)  # (B_, num_heads, WS*WS, WS*WS)
        # add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, self.num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(
            0)  # (1, num_heads, WS*WS, WS*WS)
        att = att + relative_position_bias

        if mask is not None:
            mask = mask.to(x.device)
            nW = mask.shape[0]
            att = att.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, num_windows, num_heads, WS*WS, WS*WS)
            att = att.view(B_, self.num_heads, N, N)

        att = self.softmax(att)  # (B_, num_heads, WS*WS, WS*WS)
        att = self.att_drop(att)

        x = (att @ v).transpose(1, 2).contiguous().view(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        num_feats (int): Number of input channels.
        num_heads (int): Number of attention heads.
        train_resolution (tuple[int]->(H, W)): Default input resulotion for calculate attention mask.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        use_mask (bool, optional): If True, use mask when shift_size > 0. Default: False
    """

    def __init__(self, num_feats, num_heads, train_resolution=(48, 48), window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop=0., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_mask=False):
        super(SwinTransformerBlock, self).__init__()
        self.train_resolution = train_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_mask = use_mask

        if min(train_resolution) <= shift_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.train_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(num_feats)
        self.msa = WindowAttention(num_feats, window_size=self.window_size, num_heads=num_heads,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(num_feats)
        self.mlp = MLP(num_feats, hidden_features=int(num_feats * mlp_ratio), act_layer=act_layer, drop=drop)
        # self.ffn = FeedForward(num_feats)

        if use_mask:
            att_mask = self.calculate_mask(self.train_resolution)
            self.register_buffer('att_mask', att_mask)

    def calculate_mask(self, input_resolution):
        # calculate attention mask for SW-MSA
        H, W = input_resolution
        image_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                image_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(image_mask, self.window_size)  # (num_windows, WS, WS, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        att_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (num_windows, WS*WS, WS*WS)
        att_mask = att_mask.masked_fill(att_mask != 0, float(-100.0)).masked_fill(att_mask == 0, float(0.0))
        return att_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        short_cut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows*B, WS, WS, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, WS*WS, C)

        if self.use_mask and self.shift_size > 0:
            if x_size == self.train_resolution:  # for train
                attn_windows = self.msa(x_windows, self.att_mask)
            else:
                attn_windows = self.msa(x_windows, self.calculate_mask(x_size))
        else:
            attn_windows = self.msa(x_windows, None)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, H, W)  # (B, H', W', C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, L, C)

        # FFN
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(to_token(self.ffn(to_image(self.norm2(x), x_size))))
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        num_feats (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        train_resolution (tuple[int]->(H, W)): Default input resulotion for calculate attention mask.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_mask (bool, optional): If True, use mask when shift_size > 0. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, num_feats, depth, num_heads, window_size, train_resolution=(48, 48),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop=0., drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_mask=False, use_checkpoint=False):

        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(num_feats=num_feats, num_heads=num_heads,
                                 train_resolution=train_resolution, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, drop=drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, use_mask=use_mask)
            for i in range(depth)])
        self.conv = nn.Conv2d(num_feats, num_feats, 3, 1, 1)

    def forward(self, x, x_size):
        # input (tensor): shape of (B, L, C)
        # size (tuple): (H, W)

        _input = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, torch.tensor(x_size))
            else:
                x = blk(x, x_size)

        x = self.conv(to_image(x, x_size))
        x = to_token(x)
        x = x + _input

        return x


class SwinIR(nn.Module):
    """ Swin Transformer

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_feats (int): Number of feature maps. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        train_resolution (tuple[int]->(H, W)): Default input resulotion for calculate attention mask.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        use_mask (bool, optional): If True, use mask when shift_size > 0. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, in_chans=4, num_feats=156, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=8, train_resolution=(64, 64), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 attn_drop_rate=0., drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 use_mask=False, use_checkpoint=False):
        super(SwinIR, self).__init__()
        self.window_size = window_size

        self.shallow_feature = nn.Conv2d(in_chans, num_feats, 3, 1, 1)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = BasicLayer(num_feats, depth=depths[i], num_heads=num_heads[i],
                               window_size=window_size, train_resolution=train_resolution,
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop_rate, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer, use_mask=use_mask,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(num_feats)

        self.conv_after_body = nn.Conv2d(num_feats, num_feats, 3, 1, 1)
        self.tail = nn.Conv2d(num_feats, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return 'relative_position_bias_table'

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, H, W = x.shape
        mod_pad_h = (self.window_size - H % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_feature(self, x):
        x_size = x.shape[-2:]
        x = to_token(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = to_image(x, x_size)
        x = self.conv_after_body(x)
        return x

    def forward(self, x):
        _input = x
        H, W = x.shape[-2:]
        x = self.check_image_size(x)

        x_shallow_feature = self.shallow_feature(x)
        x_deep_feature = self.forward_feature(x_shallow_feature)
        x_rec = self.tail(x_shallow_feature + x_deep_feature)

        x =  x_rec[:, :, :H, :W] + _input
        # x = torch.clamp(x, 0, 1)
        return x


if __name__ == '__main__':
    head = 5
    WS = 8
    B, H, W, C = 1, 64, 64, 30 * head
    depths = [6, 6, 6, 6, 6, 6]

    print('num_feats:', C)
    device = torch.device("cpu")

    model = SwinIR(4, C, depths=depths, num_heads=[head] * len(depths)).to(device)
    cri = nn.MSELoss().to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad is True], lr=1e-4)

    x = torch.rand((B, 4, H, W), device=device)
    gt = torch.rand((B, 4, H, W), device=device)

    model.train()
    # model.eval()
    print('input:', x.shape)
    # with torch.no_grad():
    y = model(x)
    print('output:', y.shape)

    # opt.zero_grad()
    # loss = cri(y, gt)
    # loss.backward()
    # opt.step()

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # print(model)
    torch.save(model.state_dict(), '1.pth')





