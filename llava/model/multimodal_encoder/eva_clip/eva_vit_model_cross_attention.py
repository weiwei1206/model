# --------------------------------------------------------
# Adapted from  https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except:
    from timm.layers import drop_path, to_2tuple, trunc_normal_
    
from .transformer import PatchDropout
from .rope import VisionRotaryEmbedding, VisionRotaryEmbeddingFast

if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        from torch.utils.checkpoint import checkpoint
else:
    from torch.utils.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")
    
import logging

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        drop=0.,
        subln=False,

        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

# class Attention(nn.Module):
#     def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.subln = subln
#         if self.subln:
#             self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
#             self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
#             self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
#             self.q_bias = self.k_bias = self.v_bias = None
#         else:
#             self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
#             if qkv_bias:
#                 self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
#                 self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
#         # if qkv_bias:
#         #     self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
#         #     self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
#         # else:
#         #     self.q_bias = None
#         #     self.v_bias = None

#         if window_size:
#             self.window_size = window_size
#             self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#             # cls to token & token 2 cls & cls to cls

#             # get pair-wise relative position index for each token inside the window
#             coords_h = torch.arange(window_size[0])
#             coords_w = torch.arange(window_size[1])
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#             coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#             relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
#             relative_coords[:, :, 1] += window_size[1] - 1
#             relative_coords[:, :, 0] *= 2 * window_size[1] - 1
#             relative_position_index = \
#                 torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
#             relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#             relative_position_index[0, 0:] = self.num_relative_distance - 3
#             relative_position_index[0:, 0] = self.num_relative_distance - 2
#             relative_position_index[0, 0] = self.num_relative_distance - 1

#             self.register_buffer("relative_position_index", relative_position_index)
#         else:
#             self.window_size = None
#             self.relative_position_bias_table = None
#             self.relative_position_index = None

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
#         # self.proj = nn.Linear(all_head_dim, all_head_dim)
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.xattn = xattn
#         self.xattn_drop = attn_drop

#         self.rope = rope

#     def forward(self, x, rel_pos_bias=None, attn_mask=None):
#         B, N, C = x.shape
        
#         if self.subln: 
#             # q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
#             # k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
#             # v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
#             q = self.q_proj(x)
#             k = self.k_proj(x)
#             v = self.v_proj(x)
    

#             q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
#             k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  
#             v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) 
#         else: 

#             qkv_bias = None
#             if self.q_bias is not None:
#                 qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            
#             qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#             qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
#             q, k, v = qkv[0], qkv[1], qkv[2]

#         if self.rope:
#             # slightly fast impl
#             q_t = q[:, :, 1:, :]
#             ro_q_t = self.rope(q_t)
#             q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

#             k_t = k[:, :, 1:, :]
#             ro_k_t = self.rope(k_t)
#             k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

#         if self.xattn:
#             q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
#             k = k.permute(0, 2, 1, 3)
#             v = v.permute(0, 2, 1, 3)

#             x = xops.memory_efficient_attention(
#                 q, k, v,
#                 p=self.xattn_drop,
#                 scale=self.scale,
#                 )
#             x = x.reshape(B, N, -1)
#             x = self.inner_attn_ln(x)
#             x = self.proj(x)
#             x = self.proj_drop(x)
#         else:
#             q = q * self.scale
#             attn = (q @ k.transpose(-2, -1))

#             if self.relative_position_bias_table is not None:
#                 relative_position_bias = \
#                     self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#                         self.window_size[0] * self.window_size[1] + 1,
#                         self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
#                 relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#                 attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

#             if rel_pos_bias is not None:
#                 attn = attn + rel_pos_bias.type_as(attn)

#             if attn_mask is not None:
#                 attn_mask = attn_mask.bool()
#                 attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)

#             x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#             x = self.inner_attn_ln(x)
#             x = self.proj(x)
#             x = self.proj_drop(x)
            
#         return x




# class PrefixZeroInitAttention(nn.Module):
class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        # if qkv_bias:
        #     self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        #     self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        # else:
        #     self.q_bias = None
        #     self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        # self.proj = nn.Linear(all_head_dim, all_head_dim)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope
        self.gate = torch.nn.Parameter(torch.zeros(1, num_heads, 1, 1))


    def forward(self, x, x_text, rel_pos_bias=None, attn_mask=None):     # x [32, 577, 1024]   x_text [32, 1024]
        B, N, C = x.shape  # [450, 577, 1024]
        
        if self.subln: 
            # q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            # k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            # v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
            
            q = self.q_proj(x)  # [32, 577, 1024] =  [32, 577, 1024]*[1024, 1024]
            x_k = self.k_proj(x)  # [32, 577, 1024] =  [32, 577, 1024]*[1024, 1024]
            x_v = self.v_proj(x)  # [32, 577, 1024] =  [32, 577, 1024]*[1024, 1024]
            # v = x

            if x_text.dim() != 3:
                prefix_k = self.k_proj(x_text).unsqueeze(1)  #  [32, 1, 1024]
                prefix_v = self.v_proj(x_text).unsqueeze(1)  #  [32, 1, 1024]
            else:
                prefix_k = self.k_proj(x_text)  #  [32, 1, 1024]
                prefix_v = self.v_proj(x_text)  #  [32, 1, 1024]

            k = torch.cat([prefix_k, x_k], dim=1)  # [32, 578, 1024]
            v = torch.cat([prefix_v, x_v], dim=1)  # [32, 578, 1024]

            # extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            # mask = torch.cat([extra_mask, mask], dim=-1)

            # q = q.unsqueeze(1).expand(-1, 577, -1).reshape(B, -1, self.num_heads, int(C/self.num_heads)).permute(0, 2, 1, 3)     # [450, 16, 1, 64]
            # q = q.unsqueeze(1).reshape(B, -1, self.num_heads, int(C/self.num_heads)).permute(0, 2, 1, 3)     # [450, 16, 1, 64]
            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)                                      # B, num_heads, N, C 
            k = k.reshape(B, N+prefix_k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)                                      # [450, 16, 577, 64]
            v = v.reshape(B, N+prefix_k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)                                      # [450, 16, 577, 64]
        else: 

            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        # if self.rope:
        #     # # slightly fast impl
        #     # q_t = q[:, :, 1:, :]                                     #  [450, 16, 0, 64]
        #     # ro_q_t = self.rope(q_t)                                  #  [450, 16, 576, 64]
        #     # q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)   #  [450, 16, 577, 64]

        #     k_t = k[:, :, 1:, :]                                     #  [450, 16, 576, 64]
        #     ro_k_t = self.rope(k_t)                                  #  [450, 16, 576, 64]
        #     k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)   #  [450, 16, 577, 64]
        if self.rope:
            # slightly fast impl
            #  q [32, 16, 577, 64]               
            q_t = q[:, :, 1:, :]  # [32, 16, 576, 64]  
            ro_q_t = self.rope(q_t)  # [32, 16, 576, 64]
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)  #[32, 16, 577, 64]        

            # k [32, 16, 578, 64]             
            k_t = k[:, :, 1+prefix_k.shape[1]:, :]  #  [32, 16, 577, 64]   
            ro_k_t = self.rope(k_t)  #          
            k = torch.cat((k[:, :, :1+prefix_k.shape[1], :], ro_k_t), -2).type_as(v)  

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [450, 16, 1, 577] => [batch_size, heads, query_len, key_len]
        d_k = q.size(-1)  # head_dim = 64
        attn_scores = attn_scores / (d_k ** 0.5)  # [450, 16, 1, 577]        
        # attn_weights = F.softmax(attn_scores, dim=-1)  # [450, 16, 1, 577] => [batch_size, heads, query_len, key_len]  
        # import ipdb; ipdb.set_trace()  
        attn_weights = torch.cat(
                [
                    self.gate.tanh().half() * F.softmax(attn_scores[:, :, :, :1].float(), dim=-1).type_as(q),
                    F.softmax(attn_scores[:, :, :, 1:].float(), dim=-1).type_as(q),
                ],
                dim=-1,
            )   
        
        # Weighted sum of values: [450, 16, 1, 577] x [450, 16, 577, 64]
        # attn_output = torch.matmul(attn_weights, value)  # [450, 16, 1, 64] => [batch_size, heads, query_len, head_dim]
        
        # Step 2: Broadcasting by adding dimensions
        attn_weights = attn_weights.unsqueeze(-1)  # [450, 16, 1, 577, 1]
        v = v.unsqueeze(2)   # [450, 16, 1, 577, 64]
        # x [8, 577, 1024]
        # Perform element-wise multiplication using broadcasting
        x = attn_weights * v  # Shape: [8, 16, 577, 578, 64]
        
        # Perform reduce_sum along axis=2
        x = torch.sum(x, axis=3)  # [8, 16, 577, 64]
        
        # Remove unnecessary dimensions (size 1 dimensions)
        # import ipdb; ipdb.set_trace()
        x = x.squeeze(dim=2)  # [450, 16, 577, 64] Specify the dimension to remove
        x = x.permute(0, 2, 1, 3)  # [450, 577, 16, 64]
        # x = x.view(B, N, -1)  # [450, 577, 1024]
        x = x.reshape(B, N, -1).to(torch.bfloat16) # [8, 577, 1024]
    
                    
        # x = x.reshape(B, N, -1)         #  [450, 577, 1024]
        if self.inner_attn_ln.weight.dtype != x.dtype:
            x = x.to(self.inner_attn_ln.weight.dtype)
        x = self.inner_attn_ln(x)       #  [450, 577, 1024]
        x = self.proj(x)                #  [450, 577, 1024]
        x = self.proj_drop(x) 


        # if  self.xattn:
        #     q = q.permute(0, 2, 1, 3)   #  [450, 1, 16, 64]                              B, num_heads, N, C -> B, N, num_heads, C
        #     k = k.permute(0, 2, 1, 3)   #  [450, 577, 16, 64]
        #     v = v.permute(0, 2, 1, 3)   #  [450, 577, 16, 64]

        #     x = xops.memory_efficient_attention(
        #         q, k, v,
        #         p=self.xattn_drop,
        #         scale=self.scale,
        #         )                
        #     attention = torch.matmul(q,torch.transpose(k, -1, -2))
        #     attention = torch.softmax(attention / self.norm, dim=-1)
        #     attention = torch.matmul(attention,v)
                      
        #     x = x.reshape(B, N, -1)         #  [450, 577, 1024]
        #     x = self.inner_attn_ln(x)       #  [450, 577, 1024]
        #     x = self.proj(x)                #  [450, 577, 1024]
        #     x = self.proj_drop(x)           
        # else:
        #     q = q * self.scale
        #     attn = (q @ k.transpose(-2, -1))

        #     if self.relative_position_bias_table is not None:
        #         relative_position_bias = \
        #             self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #                 self.window_size[0] * self.window_size[1] + 1,
        #                 self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        #         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #         attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

        #     if rel_pos_bias is not None:
        #         attn = attn + rel_pos_bias.type_as(attn)

        #     if attn_mask is not None:
        #         attn_mask = attn_mask.bool()
        #         attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)

        #     x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        #     x = self.inner_attn_ln(x)
        #     x = self.proj(x)
        #     x = self.proj_drop(x)
            
        return x   #  [450, 577, 1024]







class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        # if qkv_bias:
        #     self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        #     self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        # else:
        #     self.q_bias = None
        #     self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        # self.proj = nn.Linear(all_head_dim, all_head_dim)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, x_text, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape  # [450, 577, 1024]
        
        if self.subln: 
            # q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            # k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            # v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
            
            q = self.q_proj(x_text)  # [450, 577, 1024] =  [450, 577, 1024]*[1024, 1024]
            k = self.k_proj(x)  # [450, 577, 1024] =  [450, 577, 1024]*[1024, 1024]
            # v = self.v_proj(x)  # [450, 577, 1024] =  [450, 577, 1024]*[1024, 1024]
            v = x
    

            # q = q.unsqueeze(1).expand(-1, 577, -1).reshape(B, -1, self.num_heads, int(C/self.num_heads)).permute(0, 2, 1, 3)     # [450, 16, 1, 64]
            q = q.unsqueeze(1).reshape(B, -1, self.num_heads, int(C/self.num_heads)).permute(0, 2, 1, 3)     # [450, 16, 1, 64]
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)                                      # [450, 16, 577, 64]
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)                                      # [450, 16, 577, 64]
        else: 

            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            # # slightly fast impl
            # q_t = q[:, :, 1:, :]                                     #  [450, 16, 0, 64]
            # ro_q_t = self.rope(q_t)                                  #  [450, 16, 576, 64]
            # q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)   #  [450, 16, 577, 64]

            k_t = k[:, :, 1:, :]                                     #  [450, 16, 576, 64]
            ro_k_t = self.rope(k_t)                                  #  [450, 16, 576, 64]
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)   #  [450, 16, 577, 64]


        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [450, 16, 1, 577] => [batch_size, heads, query_len, key_len]
        d_k = q.size(-1)  # head_dim = 64
        attn_scores = attn_scores / (d_k ** 0.5)  # [450, 16, 1, 577]        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [450, 16, 1, 577] => [batch_size, heads, query_len, key_len]
        
        # Weighted sum of values: [450, 16, 1, 577] x [450, 16, 577, 64]
        # attn_output = torch.matmul(attn_weights, value)  # [450, 16, 1, 64] => [batch_size, heads, query_len, head_dim]
        
        # Step 2: Broadcasting by adding dimensions
        attn_weights = attn_weights.unsqueeze(-1)  # [450, 16, 1, 577, 1]
        v = v.unsqueeze(2)   # [450, 16, 1, 577, 64]
        
        # Perform element-wise multiplication using broadcasting
        x = attn_weights * v  # Shape: [450, 16, 1, 577, 64]
        
        # # Perform reduce_sum along axis=2
        # x = np.sum(x, axis=2)  # Shape: [450, 16, 64]
        
        # Remove unnecessary dimensions (size 1 dimensions)
        x = x.squeeze(dim=2)  # [450, 16, 577, 64] Specify the dimension to remove
        x = x.permute(0, 2, 1, 3)  # [450, 577, 16, 64]
        # x = x.view(B, N, -1)  # [450, 577, 1024]
        x = x.reshape(B, N, -1) 
    
                    
        # x = x.reshape(B, N, -1)         #  [450, 577, 1024]
        x = self.inner_attn_ln(x)       #  [450, 577, 1024]
        x = self.proj(x)                #  [450, 577, 1024]
        x = self.proj_drop(x) 


        # if  self.xattn:
        #     q = q.permute(0, 2, 1, 3)   #  [450, 1, 16, 64]                              B, num_heads, N, C -> B, N, num_heads, C
        #     k = k.permute(0, 2, 1, 3)   #  [450, 577, 16, 64]
        #     v = v.permute(0, 2, 1, 3)   #  [450, 577, 16, 64]

        #     x = xops.memory_efficient_attention(
        #         q, k, v,
        #         p=self.xattn_drop,
        #         scale=self.scale,
        #         )                
        #     attention = torch.matmul(q,torch.transpose(k, -1, -2))
        #     attention = torch.softmax(attention / self.norm, dim=-1)
        #     attention = torch.matmul(attention,v)
                      
        #     x = x.reshape(B, N, -1)         #  [450, 577, 1024]
        #     x = self.inner_attn_ln(x)       #  [450, 577, 1024]
        #     x = self.proj(x)                #  [450, 577, 1024]
        #     x = self.proj_drop(x)           
        # else:
        #     q = q * self.scale
        #     attn = (q @ k.transpose(-2, -1))

        #     if self.relative_position_bias_table is not None:
        #         relative_position_bias = \
        #             self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #                 self.window_size[0] * self.window_size[1] + 1,
        #                 self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        #         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #         attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

        #     if rel_pos_bias is not None:
        #         attn = attn + rel_pos_bias.type_as(attn)

        #     if attn_mask is not None:
        #         attn_mask = attn_mask.bool()
        #         attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)

        #     x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        #     x = self.inner_attn_ln(x)
        #     x = self.proj(x)
        #     x = self.proj_drop(x)
            
        return x   #  [450, 577, 1024]




# # all cross-attention
# class Block(nn.Module):

#     def __init__(self, dim, num_heads, block_id, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False,
#                  subln=False, naiveswiglu=False):
#         super().__init__()
#         self.block_id = block_id
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
#             xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
#         self.cross_attn = PrefixZeroInitAttention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
#             xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.norm3 = norm_layer(1024) # weiwei        
#         self.linear3 = nn.Linear(1280, 1024) # weiwei

#         if naiveswiglu:
#             self.mlp = SwiGLU(
#                 in_features=dim, 
#                 hidden_features=mlp_hidden_dim, 
#                 subln=subln,
#                 norm_layer=norm_layer,
#             )
#         else:
#             self.mlp = Mlp(
#                 in_features=dim, 
#                 hidden_features=mlp_hidden_dim, 
#                 act_layer=act_layer,
#                 subln=subln,
#                 drop=drop
#             )

#         if init_values is not None and init_values > 0:
#             self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
#             self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
#         else:
#             self.gamma_1, self.gamma_2 = None, None

#         self.postnorm = postnorm

#     def forward(self, x, x_text, rel_pos_bias=None, attn_mask=None):
#         if self.gamma_1 is None:
#             if self.postnorm:
#                 x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
#                 x = x + self.drop_path(self.norm2(self.mlp(x)))
#             else:
#                 x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) #
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) #                
#                 x = x + self.drop_path(self.mlp(self.norm2(x))) 
#         else:
#             if self.postnorm:
#                 x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
#                 x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
#             else:
#                 x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
#                 x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
#         return x



# last block cross-attention
class Block(nn.Module):

    def __init__(self, dim, num_heads, block_id, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False,
                 subln=False, naiveswiglu=False):
        super().__init__()
        self.block_id = block_id
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
            xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        # self.cross_attn = PrefixZeroInitAttention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
        #     xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm3 = norm_layer(1024) # weiwei        
        self.linear3 = nn.Linear(1280, 1024) # weiwei

        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                act_layer=act_layer,
                subln=subln,
                drop=drop
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, x_text, rel_pos_bias=None, attn_mask=None, is_last_block=False):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                # cross-attentions
                if is_last_block:
                    x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(x_text), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                # import ipdb; ipdb.set_trace()
                if is_last_block:
                    if self.norm3.weight.dtype != x_text.dtype:
                        x_text = x_text.to(self.norm3.weight.dtype)
                x = x + self.drop_path(self.attn(self.norm1(x), self.norm3(x_text), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) #
                # # cross-attentions
                # if is_last_block:
                #     if self.mlp.weight.dtype != x_text.dtype:
                #         x_text = x_text.to(self.mlp.weight.dtype)
                #     x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(x_text), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) #                
                x = x + self.drop_path(self.mlp(self.norm2(x))) 
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                # cross-attentions
                if is_last_block:
                    x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                # cross-attentions
                if is_last_block:
                    x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
                
        return x




# # last six block cross-attention
# class Block(nn.Module):

#     def __init__(self, dim, num_heads, block_id, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False,
#                  subln=False, naiveswiglu=False):
#         super().__init__()
#         self.block_id = block_id
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
#             xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
#         self.cross_attn = CrossAttention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
#             xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.norm3 = norm_layer(1024) # weiwei        
#         self.linear3 = nn.Linear(1280, 1024) # weiwei

#         if naiveswiglu:
#             self.mlp = SwiGLU(
#                 in_features=dim, 
#                 hidden_features=mlp_hidden_dim, 
#                 subln=subln,
#                 norm_layer=norm_layer,
#             )
#         else:
#             self.mlp = Mlp(
#                 in_features=dim, 
#                 hidden_features=mlp_hidden_dim, 
#                 act_layer=act_layer,
#                 subln=subln,
#                 drop=drop
#             )

#         if init_values is not None and init_values > 0:
#             self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
#             self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
#         else:
#             self.gamma_1, self.gamma_2 = None, None

#         self.postnorm = postnorm

#     def forward(self, x, x_text, rel_pos_bias=None, attn_mask=None):
#         if self.gamma_1 is None:
#             if self.postnorm:
#                 x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
#                 x = x + self.drop_path(self.norm2(self.mlp(x)))
#             else:
#                 x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) #
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) #                
#                 x = x + self.drop_path(self.mlp(self.norm2(x))) 
#         else:
#             if self.postnorm:
#                 x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
#                 x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
#             else:
#                 x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
#                 x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
#         return x




# # Even-numbered cross-attention
# class Block(nn.Module):

#     def __init__(self, dim, num_heads, block_id, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False,
#                  subln=False, naiveswiglu=False):
#         super().__init__()
#         self.block_id = block_id
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
#             xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
#         self.cross_attn = CrossAttention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
#             xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.norm3 = norm_layer(1024) # weiwei        
#         self.linear3 = nn.Linear(1280, 1024) # weiwei

#         if naiveswiglu:
#             self.mlp = SwiGLU(
#                 in_features=dim, 
#                 hidden_features=mlp_hidden_dim, 
#                 subln=subln,
#                 norm_layer=norm_layer,
#             )
#         else:
#             self.mlp = Mlp(
#                 in_features=dim, 
#                 hidden_features=mlp_hidden_dim, 
#                 act_layer=act_layer,
#                 subln=subln,
#                 drop=drop
#             )

#         if init_values is not None and init_values > 0:
#             self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
#             self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
#         else:
#             self.gamma_1, self.gamma_2 = None, None

#         self.postnorm = postnorm

#     def forward(self, x, x_text, rel_pos_bias=None, attn_mask=None):
#         if self.gamma_1 is None:
#             if self.postnorm:
#                 x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
#                 x = x + self.drop_path(self.norm2(self.mlp(x)))
#             else:
#                 x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) #
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) #                
#                 x = x + self.drop_path(self.mlp(self.norm2(x))) 
#         else:
#             if self.postnorm:
#                 x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
#                 x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
#             else:
#                 x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
#                 # cross-attentions
#                 x = x + self.drop_path(self.cross_attn(self.norm1(x), self.norm3(self.linear3(x_text)), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)) # 
#                 x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
#         return x
























class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class EVAVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, patch_dropout=0.,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, rope=False,
                 use_mean_pooling=True, init_scale=0.001, grad_checkpointing=False, xattn=False, postnorm=False,
                 pt_hw_seq_len=16, intp_freq=False, naiveswiglu=False, subln=False):
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
                # patch_dropout=patch_dropout
            )
        else: 
            self.rope = None

        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, block_id=i, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                xattn=xattn, rope=self.rope, postnorm=postnorm, subln=subln, naiveswiglu=naiveswiglu)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, 1280) if num_classes > 0 else nn.Identity()
        
        # lock head
        # for param in self.head.parameters():
        #     param.requires_grad = False

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.grad_checkpointing = grad_checkpointing
        self.unlocked_groups = 999
        
        # self.set_block_gradients([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
        
        
    def set_block_gradients(self, blocks_to_train):
        for i, block in enumerate(self.blocks):
            for param in block.parameters():
                param.requires_grad = (i in blocks_to_train)
        
        
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher
    
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        # assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        # for param in self.parameters():
        #     param.requires_grad = False
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self)
        else:
            self.unlocked_groups = unlocked_groups
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.group_matcher()
            gparams = group_parameters(self, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self, gmodules)

        # if not unlocked_groups:
        #     # lock full model except head
        #     for name,param in self.named_parameters():
        #         if 'head' not in name:
        #             param.requires_grad = False
        # else:
        #     image_fix_num = "blocks.{}".format(int(unlocked_groups))
        #     for name, parms in self.named_parameters():
        #         if image_fix_num in name:
        #             break
        #         print(f'fixed params: {name}.')
        #         logging.info(f'fixed params: {name}.')
        #         parms.requires_grad_(False)

        #         if "cls_token" not in name :
        #             print("fixed params:", name)
        #         logging.info(f'fixed params: {name}.')
        #         parms.requires_grad_(False)
        
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, x_text, return_all_features=True):  # [32, 3, 336, 336]
        x = self.patch_embed(x)                                   # [32, 3, 336, 336]
        batch_size, seq_len, _ = x.size()                    

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)    # [450, 1, 1024]          stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)                     # [450, 576, 1024]
        if self.pos_embed is not None:
            x = x + self.pos_embed                                # [450, 577, 1024]
        x = self.pos_drop(x)                                      # [450, 577, 1024]

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        
        start_layer = len(self.blocks) +1  - self.unlocked_groups
        for i, blk in enumerate(self.blocks):                                 # BLOCK
            # if self.grad_checkpointing and i > start_layer:
            is_last_block = True
            # is_last_block = (i == len(self.blocks) - 1) 
            if self.grad_checkpointing:
                x = checkpoint(blk, x, x_text, (rel_pos_bias,), use_reentrant=False, is_last_block=is_last_block)  # [450, 577, 1024]
                # x = checkpoint(blk, x, (rel_pos_bias,))
            else:
                x = blk(x, x_text, rel_pos_bias=rel_pos_bias, is_last_block=is_last_block)

        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x


    def forward(self, x, x_text, return_all_features=True, output_hidden_states=False):  # x [450, 3, 336, 336]
        if return_all_features:
            return self.forward_features(x, x_text, return_all_features)
        x = self.forward_features(x, x_text)  # [450, 1024]
        # x = self.head(x)
        return x
