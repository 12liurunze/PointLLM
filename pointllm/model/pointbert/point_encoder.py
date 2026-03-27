import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .dvae import Group
from .dvae import Encoder
from . import misc
from .logger import print_log
from collections import OrderedDict

from .checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class SlotPointTokenizer(nn.Module):
    """Learnable slot tokenizer for point clouds (non-grouping)."""
    def __init__(self, point_dims, trans_dim, num_slots, num_heads=8):
        super().__init__()
        self.num_slots = num_slots
        self.point_proj = nn.Sequential(
            nn.Linear(point_dims, trans_dim),
            nn.GELU(),
            nn.Linear(trans_dim, trans_dim),
        )
        self.slot_embed = nn.Parameter(torch.randn(1, num_slots, trans_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=trans_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(trans_dim)
        self.slot_mlp = nn.Sequential(
            nn.Linear(trans_dim, trans_dim * 4),
            nn.GELU(),
            nn.Linear(trans_dim * 4, trans_dim),
        )
        self.out_norm = nn.LayerNorm(trans_dim)

    def forward(self, pts):
        point_tokens = self.point_proj(pts)  # B N C
        slots = self.slot_embed.expand(pts.shape[0], -1, -1)
        attn_out, attn_weights = self.cross_attn(
            query=slots,
            key=point_tokens,
            value=point_tokens,
            need_weights=True,
        )
        slots = self.slot_norm(slots + attn_out)
        slots = self.out_norm(slots + self.slot_mlp(slots))
        # center by attention-weighted xyz for positional embedding
        centers = torch.matmul(attn_weights, pts[:, :, :3].contiguous())  # B G 3
        return slots, centers


class PointTransformer(nn.Module):
    def __init__(self, config, use_max_pool=True):
        super().__init__()
        self.config = config
        
        self.use_max_pool = use_max_pool # * whethet to max pool the features of different tokens

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.point_dims = config.point_dims
        self.tokenizer_type = getattr(config, "tokenizer_type", "group")

        self.encoder_dims = config.encoder_dims
        if self.tokenizer_type == "group":
            # default PointBERT grouping tokenizer
            self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
            self.encoder = Encoder(encoder_channel=self.encoder_dims, point_input_dims=self.point_dims)
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        elif self.tokenizer_type == "fps_mlp":
            # alternative tokenizer: FPS center sampling + point-wise MLP tokenization (no local grouping)
            self.group_divider = None
            self.encoder = None
            self.reduce_dim = nn.Sequential(
                nn.Linear(self.point_dims, self.encoder_dims),
                nn.GELU(),
                nn.Linear(self.encoder_dims, self.trans_dim),
            )
        elif self.tokenizer_type == "avg_pool_mlp":
            # alternative tokenizer: adaptive pooling on raw points + MLP tokenization (no FPS/KNN grouping)
            self.group_divider = None
            self.encoder = None
            self.reduce_dim = nn.Sequential(
                nn.Linear(self.point_dims, self.encoder_dims),
                nn.GELU(),
                nn.Linear(self.encoder_dims, self.trans_dim),
            )
        elif self.tokenizer_type == "slot_attn":
            # alternative tokenizer: learnable slots cross-attend to all points
            self.group_divider = None
            self.encoder = None
            self.reduce_dim = None
            self.slot_tokenizer = SlotPointTokenizer(
                point_dims=self.point_dims,
                trans_dim=self.trans_dim,
                num_slots=self.num_group,
                num_heads=max(1, min(self.num_heads, self.trans_dim // 32)),
            )
        else:
            raise ValueError(f"Unsupported tokenizer_type: {self.tokenizer_type}")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def load_checkpoint(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.point_encoder.'):
                state_dict[k.replace('module.point_encoder.', '')] = v

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            # * print successful loading
            print_log("PointBERT's weights are successfully loaded from {}".format(bert_ckpt_path), logger='Transformer')

    def forward(self, pts):
        if self.tokenizer_type == "group":
            # divide the point cloud in the same form. This is important for PointBERT pretraining checkpoints
            neighborhood, center = self.group_divider(pts)
            # encode the grouped cloud blocks
            group_input_tokens = self.encoder(neighborhood)  # B G N
            group_input_tokens = self.reduce_dim(group_input_tokens)
        elif self.tokenizer_type == "fps_mlp":
            # use FPS centers as tokens directly (no local grouping)
            center = misc.fps(pts[:, :, :3].contiguous(), self.num_group)  # B G 3
            # approximate token features via nearest-point retrieval from sampled centers
            dist = torch.cdist(center, pts[:, :, :3].contiguous())  # B G N
            nearest_idx = dist.argmin(dim=-1)  # B G
            nearest_idx_expand = nearest_idx.unsqueeze(-1).expand(-1, -1, self.point_dims)
            sampled_points = torch.gather(pts, 1, nearest_idx_expand)  # B G C
            group_input_tokens = self.reduce_dim(sampled_points)
        else:  # avg_pool_mlp
            if self.tokenizer_type == "avg_pool_mlp":
                # adaptive pooling over raw points to fixed token count (order-sensitive but lightweight)
                pooled_points = F.adaptive_avg_pool1d(pts.transpose(1, 2), self.num_group).transpose(1, 2)
                center = pooled_points[:, :, :3]
                group_input_tokens = self.reduce_dim(pooled_points)
            else:  # slot_attn
                group_input_tokens, center = self.slot_tokenizer(pts)

        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x) # * B, G + 1(cls token)(513), C(384)
        if not self.use_max_pool:
            return x
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1).unsqueeze(1) # * concat the cls token and max pool the features of different tokens, make it B, 1, C
        return concat_f # * B, 1, C(384 + 384)
