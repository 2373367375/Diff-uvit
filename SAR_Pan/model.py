import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from einops import rearrange, repeat
import en_transformer


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class conv_3_3(nn.Module):
    def __init__(self, out_channels):
        super(conv_3_3, self).__init__()
        mid_channels = out_channels//2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.unsqueeze(dim=0)
        x = x.unsqueeze(dim=0)
        x = x.to(torch.float32)  ####
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.squeeze(dim=0)
        return x

class Lidar_Specific_VisionTransformer(nn.Module):
    def __init__(self, img_size, input_channels, patch_size, dim, depth, heads, mlp_dim, dropout, dim_head, pool='cls'):
        super(Lidar_Specific_VisionTransformer, self).__init__()

        num_patches = (img_size // patch_size) ** 2
        patch_dim = input_channels * patch_size ** 2

        assert pool in {'cls', 'mean'}
        self.pool = pool

        self.patch_embedding = nn.Conv2d(input_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.to_cls_token = nn.Identity()
        self.transformer_encoder = en_transformer.Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                                                              mlp_head=mlp_dim, dropout=dropout,
                                                              num_channel=num_patches)

        # self.MLP = nn.Sequential(
        #     nn.Linear(dim, 64),
        #     nn.BatchNorm1d(64),# 输入维度到128维
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.BatchNorm1d(32),# 128维到64维
        #     nn.ReLU(),)

    def forward(self, data_HSI):
        data_HSI = self.patch_embedding(data_HSI)  # (batch_size, dim, num_patches_h, num_patches_w)
        data_HSI = rearrange(data_HSI, 'b d h w -> b (h w) d')  # (batch_size, num_patches, dim)
        b, n, _ = data_HSI.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        data_HSI = torch.cat((cls_tokens, data_HSI), dim=1)
        positional_embedding = self.positional_embedding[:, :(n+1)]
        data_HSI = data_HSI + positional_embedding
        data_HSI = self.dropout(data_HSI)

        data_HSI = self.transformer_encoder(data_HSI)
        # global average pooling or cls
        data_HSI = data_HSI.mean(dim=1) if self.pool == 'mean' else data_HSI[:, 0]
        # data_HSI = self.MLP(data_HSI)
        return data_HSI

class HSI_Specific_VisionTransformer(nn.Module):
    def __init__(self, img_size, input_channels, patch_size, dim, depth, heads, mlp_dim, dropout, dim_head, pool='cls'):
        super(HSI_Specific_VisionTransformer, self).__init__()

        num_patches = (img_size // patch_size) ** 2
        patch_dim = input_channels * patch_size ** 2

        assert pool in {'cls', 'mean'}
        self.pool = pool

        self.patch_embedding = nn.Conv2d(input_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.to_cls_token = nn.Identity()
        self.transformer_encoder = en_transformer.Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                                                              mlp_head=mlp_dim, dropout=dropout,
                                                              num_channel=num_patches)

        # self.MLP = nn.Sequential(
        #     nn.Linear(dim, 64),
        #     nn.BatchNorm1d(64),# 输入维度到128维
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.BatchNorm1d(32),# 128维到64维
        #     nn.ReLU(),)

    def forward(self, data_HSI):
        data_HSI = self.patch_embedding(data_HSI)  # (batch_size, dim, num_patches_h, num_patches_w)
        data_HSI = rearrange(data_HSI, 'b d h w -> b (h w) d')  # (batch_size, num_patches, dim)
        b, n, _ = data_HSI.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        data_HSI = torch.cat((cls_tokens, data_HSI), dim=1)
        positional_embedding = self.positional_embedding[:, :(n+1)]
        data_HSI = data_HSI + positional_embedding
        data_HSI = self.dropout(data_HSI)

        data_HSI = self.transformer_encoder(data_HSI)
        # global average pooling or cls
        data_HSI = data_HSI.mean(dim=1) if self.pool == 'mean' else data_HSI[:, 0]
        # data_HSI = self.MLP(data_HSI)
        return data_HSI


class Shared_VisionTransformer(nn.Module):
    def __init__(self, img_size, input_channels, patch_size, dim, depth, heads, mlp_dim, dropout, dim_head, pool='cls'):
        super(Shared_VisionTransformer, self).__init__()

        num_patches = (img_size // patch_size) ** 2
        patch_dim = input_channels * patch_size ** 2

        assert pool in {'cls', 'mean'}
        self.pool = pool

        self.patch_embedding = nn.Conv2d(input_channels, dim, 3, 1,1)
        self.positional_embedding = nn.Parameter(torch.randn(1, 25 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.to_cls_token = nn.Identity()
        self.transformer_encoder = en_transformer.Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                                                              mlp_head=mlp_dim, dropout=dropout,
                                                              num_channel=num_patches)

        # self.MLP = nn.Sequential(
        #     nn.Linear(dim*2, 64),
        #     nn.BatchNorm1d(64),# 输入维度到128维
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.BatchNorm1d(32),# 128维到64维
        #     nn.ReLU(),)
        self.flatten = nn.Flatten(1,-1)
        self.MLP = nn.Linear(128, 2)
    def forward(self, data_HSI, data_Lidar):
        data_HSI = data_HSI['T2_f_step0_0_patch']
        data_HSI = self.patch_embedding(data_HSI)  # (batch_size, dim, num_patches_h, num_patches_w)
        data_HSI = rearrange(data_HSI, 'b d h w -> b (h w) d')  # (batch_size, num_patches, dim)
        b, n, _ = data_HSI.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        data_HSI = torch.cat((cls_tokens, data_HSI), dim=1)
        positional_embedding = self.positional_embedding[:, :(n+1)]
        data_HSI = data_HSI + positional_embedding
        data_HSI = self.dropout(data_HSI)

        data_HSI = self.transformer_encoder(data_HSI)
        # global average pooling or cls
        data_HSI = data_HSI.mean(dim=1) if self.pool == 'mean' else data_HSI[:, 0]
        # proj = self.MLP(data_HSI)

        # data_Lidar = self.patch_embedding(data_Lidar)  # (batch_size, dim, num_patches_h, num_patches_w)
        # data_Lidar = rearrange(data_Lidar, 'b d h w -> b (h w) d')  # (batch_size, num_patches, dim)
        # b, n, _ = data_Lidar.shape
        #
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # data_Lidar = torch.cat((cls_tokens, data_Lidar), dim=1)
        # positional_embedding = self.positional_embedding[:, :(n+1)]
        # data_Lidar = data_Lidar + positional_embedding
        # data_Lidar = self.dropout(data_Lidar)
        #
        # data_Lidar = self.transformer_encoder(data_Lidar)
        # global average pooling or cls
        # data_Lidar = data_Lidar.mean(dim=1) if self.pool == 'mean' else data_Lidar[:, 0]
        data = data_HSI
        data = self.flatten(data)
        data = self.MLP(data)
        return data

class Text_encoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()
        self.context_length = context_length


        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width) #在模型中归一化最终的输出
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim)) #用于将Transformer模型的输出投影到一个更低维度的表示,transformer_width是输入维度，而embed_dim是输出维度
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) #调整模型输出的尺度
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text).type(torch.float32)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(torch.float32)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(torch.float32)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class LPCnet(nn.Module):
     def __init__(self, patchsize_HSI = 11,input_channels = 15, patch_size = 1, dim = 32,depth = 3,heads = 4,mlp_dim = 32,dropout = 0,dim_head = 16):
         super(LPCnet, self).__init__()
         self.lidar_conv = nn.Sequential(
             nn.Conv2d(1,32,3,1,1),
             nn.ReLU(),
             nn.Conv2d(32, input_channels, 3, 1, 1),
         )
         self.Shared_vision_net = Shared_VisionTransformer(img_size=patchsize_HSI,input_channels=input_channels,patch_size=patch_size,
                                                    dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, dim_head=dim_head)
         self.HSI_vision_net = HSI_Specific_VisionTransformer(img_size=patchsize_HSI,input_channels=input_channels,patch_size=patch_size,
                                                    dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, dim_head=dim_head)
         self.Lidar_vision_net = Lidar_Specific_VisionTransformer(img_size=patchsize_HSI,input_channels=input_channels,patch_size=patch_size,
                                                    dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, dim_head=dim_head)
         self.text_net = Text_encoder(embed_dim=512,  ## 输出维度，可更改,模型最后一层就得改
                                 context_length=77,
                                 vocab_size=49408,
                                 transformer_width=512,
                                 transformer_heads=8,
                                 transformer_layers=12)
         self.shared_projector_vision = nn.Sequential(nn.Linear(32, 512))
         self.hsi_projector_vision = nn.Sequential(nn.Linear(32, 512))
         self.lidar_projector_vision = nn.Sequential(nn.Linear(32, 512))

         self.mlp = nn.Linear(32*3,15)

         self.softmax = nn.Softmax(dim=1)
     def forward(self,patch_hsi,patch_lidar, text):
         with torch.no_grad():
             text = clip.tokenize(text).cuda(1)  #
             text_features = self.text_net(text)

         patch_lidar = self.lidar_conv(patch_lidar)

         Shared_vision_features = self.Shared_vision_net(patch_hsi, patch_lidar)
         HSI_vision_features = self.HSI_vision_net(patch_hsi)
         Lidar_vision_features = self.Lidar_vision_net(patch_lidar)
         proj_Shared_vision_features = self.shared_projector_vision(Shared_vision_features)
         proj_HSI_vision_features = self.hsi_projector_vision(HSI_vision_features)
         proj_Lidar_vision_features = self.lidar_projector_vision(Lidar_vision_features)

         vision_features = torch.cat((Shared_vision_features, HSI_vision_features, Lidar_vision_features),dim = 1)
         logit = self.softmax(self.mlp(vision_features))


         return proj_Shared_vision_features, proj_HSI_vision_features, proj_Lidar_vision_features, text_features, logit