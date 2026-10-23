# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Definition of Infinity transformer model.
"""

import math
import os
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any
import json

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
import numpy as np
from torch.nn.attention.flex_attention import flex_attention

import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_fused_op_installed, SelfAttnBlock, FastRMSNorm
from infinity.models.rope import precompute_rope4d_freqs_grid
from infinity.models.flex_attn_mask import build_flex_attn_func
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index, get_activated_h_div_w_templates
from infinity.models.apg import normalized_guidance
# KV pooling is handled inside basic.py's SelfAttention
from infinity.utils.sequence_parallel import sp_split_sequence_by_dim, sp_gather_sequence_by_dim, SequenceParallelManager as sp_manager

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C

class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None, scale_ind=None, context_info=None, last_repetition_step=True, ref_text_scale_inds=[], cond_only=False, uncond_only=False, mlp_keep_idx=None):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_repetition_step, ref_text_scale_inds, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_repetition_step, ref_text_scale_inds, cond_only=cond_only, uncond_only=uncond_only, mlp_keep_idx=mlp_keep_idx)
        return h

def get_timestep_embedding(dim, timesteps=1000, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    assert dim % 2 == 0, "dimension must be even number"
    half = dim // 2
    timesteps = torch.arange(timesteps, dtype=torch.float32)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

class Infinity(nn.Module):
    def __init__(
        self, vae_local,
        arch='qwen',                         # var or qwen
        qwen_qkvo_bias=False,               # qwen qwen_qkvo_bias
        text_channels=0, text_maxlen=0,     # text-cond generation
        embed_dim=1024, depth=16, 
        num_key_value_heads=-1,
        num_heads=16, mlp_ratio=4.,   # model's architecture
        norm_eps=1e-6, rms_norm=False,      # norm layer
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        drop_path_rate=0.1,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        top_p=0.0, 
        top_k=0.0,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        add_lvl_embeding_on_first_block=1,
        num_of_label_value=2,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        apply_spatial_patchify = 0,
        inference_mode=False,
        other_args=None,
    ):
        super().__init__()
        # set hyperparameters
        self.C = embed_dim
        self.vae_embed_dim = vae_local.codebook_dim
        self.detail_scale_min_tokens = other_args.detail_scale_min_tokens
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.codebook_dim * 4
        else:
            self.d_vae = vae_local.codebook_dim
        self.other_args = other_args
        self.mask_type = other_args.mask_type
        self.context_frames = other_args.context_frames
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.video_frames)
        self.num_of_label_value = num_of_label_value
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * self.num_of_label_value) if self.num_of_label_value else vae_local.vocab_size
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.image_batch_size = other_args.image_batch_size
        self.video_batch_size = other_args.video_batch_size
        self.arch = arch
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = get_activated_h_div_w_templates(train_h_div_w_list, self.h_div_w_templates)
        self.video_frames = video_frames


        assert add_lvl_embeding_on_first_block in [0,1]
        self.add_lvl_embeding_on_first_block = add_lvl_embeding_on_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        self.image_scale_repetition = json.loads(other_args.image_scale_repetition)
        self.video_scale_repetition = json.loads(other_args.video_scale_repetition)
        print(f'arch: {arch}, self.pn: {self.pn}, self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_on_first_block: {self.add_lvl_embeding_on_first_block}, \
            self.num_of_label_value: {self.num_of_label_value}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw} \
            self.train_h_div_w_list: {self.train_h_div_w_list}, self.image_scale_repetition: {self.image_scale_repetition}, self.video_scale_repetition: {self.video_scale_repetition}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        
        # [inp & position embedding]
        self.norm0_cond = nn.Identity()
        self.selecting_idx = None
        self.num_classes = 0
        self.D = self.C
        
        cfg_uncond = torch.empty(512, self.Ct5)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(0)
        torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
        cfg_uncond /= self.Ct5 ** 0.5
        if rand_uncond:
            self.register_buffer('cfg_uncond', cfg_uncond)
        else:
            self.cfg_uncond = nn.Parameter(cfg_uncond)
        
        if other_args.simple_text_proj:
            self.text_norm = nn.Identity()
            self.text_proj = nn.Linear(self.Ct5, self.D)
        else:
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )
        self.sos_token = nn.Parameter(torch.empty(1, 1, self.D))
        
        if self.rope2d_each_sa_layer:
            if other_args.rope_type == '4d':
                tmp_h_div_w_template = self.train_h_div_w_list[0]
                scales_in_one_clip = self.dynamic_resolution_h_w[tmp_h_div_w_template][self.pn]['scales_in_one_clip']
                max_video_scales = self.dynamic_resolution_h_w[tmp_h_div_w_template][self.pn]['max_video_scales']
                if other_args.dynamic_scale_schedule == 'infinity_star_interact':
                    max_scales = 1000
                else:
                    max_scales = sum(self.image_scale_repetition) + sum(self.video_scale_repetition) * (max_video_scales//scales_in_one_clip-1)
                    max_scales = max(max_scales, max_video_scales)
                rope2d_freqs_grid = precompute_rope4d_freqs_grid(dim=self.C//self.num_heads,
                                                                 pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                                                                 activated_h_div_w_templates=self.train_h_div_w_list,
                                                                 steps_per_frame=other_args.steps_per_frame,
                                                                 max_scales=max_scales+10,
                                                                 max_frames=int(self.video_frames/other_args.temporal_compress_rate+1),
                                                                 max_height=1800 // 8, max_width=1800 // 8,
                                                                 text_maxlen=self.text_maxlen,
                                                                 pn=self.pn,
                                                                 args=other_args,)
            else:
                raise ValueError(f'self.rope_type == {self.rope_type} unsupported!')
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)  
        if self.arch == 'qwen':
            self.norm_hidden_sates = FastRMSNorm(self.C)
        else:
            raise ValueError(f'arch={self.arch} not implemented')
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        if self.use_flex_attn:
            self.flex_attention = torch.compile(flex_attention)

        self.unregistered_blocks = []
        for _ in range(depth):
            block = SelfAttnBlock(
                embed_dim=self.C, 
                cond_dim=self.D,
                num_heads=num_heads, 
                num_key_value_heads=num_key_value_heads, 
                mlp_ratio=mlp_ratio,
                use_flex_attn=use_flex_attn, 
                pad_to_multiplier=pad_to_multiplier, 
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                mask_type=other_args.mask_type, 
                context_frames=other_args.context_frames, 
                steps_per_frame=other_args.steps_per_frame,
                arch=self.arch,
                qwen_qkvo_bias=qwen_qkvo_bias,
                inject_sync=other_args.inject_sync,
            )
            # block.bfloat16()
            self.unregistered_blocks.append(block)
        
        # [head]
        self.head = nn.Linear(self.C, self.other_args.detail_scale_dim*self.other_args.num_of_label_value)
        if self.other_args.use_two_stage_lfq:
            self.semantic_head2 = nn.Linear(self.C, self.other_args.semantic_scale_dim*self.other_args.num_of_label_value)
            
        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n',
            end='\n\n', flush=True
        )
        
    def get_loss_acc(self, x_BLC, sequece_packing_scales, gt):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        if self.arch == 'qwen':
            x_BLC = self.norm_hidden_sates(x_BLC)
        
        with torch.amp.autocast('cuda', enabled=False):
            x_BLC = x_BLC.float()
            logits_full = self.head(x_BLC)
            if self.other_args.use_two_stage_lfq:
                logits_semantic_full = self.semantic_head2(x_BLC)
                global_token_ptr, global_scale_ptr = 0, 0
                loss_list, acc_list = [], []
                for i in range(len(sequece_packing_scales)):
                    for j in range(len(sequece_packing_scales[i])):
                        pt, ph, pw = sequece_packing_scales[i][j]
                        mul_pt_ph_pw = pt * ph * pw
                        if ph * pw >= self.detail_scale_min_tokens:
                            logits = logits_full[:,global_token_ptr:global_token_ptr+mul_pt_ph_pw]
                        else:
                            logits = logits_semantic_full[:,global_token_ptr:global_token_ptr+mul_pt_ph_pw]
                        logits = logits.reshape(x_BLC.shape[0], mul_pt_ph_pw, -1, self.other_args.num_of_label_value)
                        logits = logits.permute(0,3,1,2) # [1, mul_pt_ph_pw, d, num_of_label_value] -> [1, num_of_label_value, mul_pt_ph_pw, d]
                        # gt[global_scale_ptr]: [1, mul_pt_ph_pw, d]
                        loss_this_scale = F.cross_entropy(logits, gt[global_scale_ptr], reduction='none').mean(-1)[0] # [mul_pt_ph_pw]
                        acc_this_scale = (logits.argmax(1) == gt[global_scale_ptr]).float().mean(-1)[0] # [mul_pt_ph_pw]

                        loss_list.append(loss_this_scale)
                        acc_list.append(acc_this_scale)
                        global_scale_ptr += 1
                        global_token_ptr += mul_pt_ph_pw
                loss_list = torch.cat(loss_list)
                acc_list = torch.cat(acc_list)
            else:
                gt = torch.cat(gt, 1) # [B, L, d]
                logits = logits_full
                logits = logits.reshape(x_BLC.shape[0], x_BLC.shape[1], -1, self.other_args.num_of_label_value)
                logits = logits.permute(0,3,1,2) # [B, num_of_label_value, L, d]
                if self.other_args.num_of_label_value > 1:
                    loss_list = F.cross_entropy(logits, gt, reduction='none').mean(-1)[0] # [L]
                    acc_list = (logits.argmax(1) == gt).float().mean(-1)[0] # [L]
                elif self.other_args.num_of_label_value == 1:
                    loss_list = torch.nn.functional.mse_loss(logits.squeeze(1), gt[global_scale_ptr], reduction='none').mean(-1)[0] # [L]
                    acc_list = loss_list
            return loss_list, acc_list
    
    def get_logits_during_infer(self, x_BLC, is_semantic_scale):
        if self.arch == 'qwen':
            x_BLC = self.norm_hidden_sates(x_BLC)
        with torch.amp.autocast('cuda', enabled=False):
            x_BLC = x_BLC.float()
            if self.other_args.use_two_stage_lfq:
                if is_semantic_scale:
                    logits = self.semantic_head2(x_BLC)
                else:
                    logits = self.head(x_BLC)
            else:
                logits = self.head(x_BLC)
        return logits
    
    def pick_visual_tokens(
        self,
        x_BLC,
        sequece_packing_scales,
        visual_tokens_len,
        args,
    ):
        visual_tokens = x_BLC[:,:visual_tokens_len]
        return visual_tokens
    
    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC: torch.Tensor,
        visual_rope_cache = None,
        sequece_packing_scales = None, # [[(1,1,1)->(5,5,5)], [(1,1,1)->(10,10,10)]] 1LC
        super_scale_lengths = None,
        super_querysid_super_refsid = None,
        other_info_by_scale = None,
        gt_BL = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """
        
        x_BLC= x_BLC.float()       # input should be float32
        B = x_BLC.shape[0]
        cond_BD_or_gss, ca_kv = None, None

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # 12 kv_compact, lens

            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact[0, 0] += must_on_graph
            # drop cond
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le
            
            visual_tokens_len = x_BLC.shape[1]
            # forms prefix_tokens
            kv_compact = self.text_norm(kv_compact)
            kv_compact = self.text_proj(kv_compact).contiguous()
            x_BLC = self.word_embed(self.norm0_ve(x_BLC)) # norm0_ve is Identity
            x_BLC = torch.cat((x_BLC, kv_compact.unsqueeze(0)), dim=1)

            if self.other_args.train_with_var_seq_len:
                pad_seq_len = int(np.ceil(x_BLC.shape[1]/self.pad_to_multiplier))*self.pad_to_multiplier - x_BLC.shape[1]
            else:
                pad_seq_len = self.other_args.train_max_token_len - x_BLC.shape[1]
            if pad_seq_len > 0:
                x_BLC = F.pad(x_BLC, (0, 0, 0, pad_seq_len), value=0.0)

            # valid_sequence_ratio = 1 - pad_seq_len / self.other_args.train_max_token_len
            valid_sequence_ratio = 1 - pad_seq_len / x_BLC.shape[1]
            assert self.use_flex_attn
            attn_bias_or_two_vector = None
        
        attn_fn = build_flex_attn_func(
            flex_attention=self.flex_attention,
            seq_l=x_BLC.shape[1],
            prefix_lens=lens,
            args=self.other_args,
            device=x_BLC.device,
            batch_size=B,
            heads=None,
            pad_seq_len=pad_seq_len,
            sequece_packing_scales=sequece_packing_scales,
            super_scale_lengths=super_scale_lengths,
            super_querysid_super_refsid=super_querysid_super_refsid,
        )

        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(x_BLC.device)
        rope_cache_list = [visual_rope_cache]
        for i in range(len(lens)):
            rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[i]])
        rope_cache = torch.cat(rope_cache_list, dim=4)
        if pad_seq_len > 0:
            rope_cache = F.pad(rope_cache, (0,0,0,pad_seq_len), 'constant', 0.)
        assert rope_cache.shape[4] == x_BLC.shape[1], f'{rope_cache.shape[4]} != {x_BLC.shape[1]}'
        # [2. block loop]
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training

        if sp_manager.sp_on():
            # [B, raw_L, C] --> [B, raw_L/sp_size, C]
            x_BLC = sp_split_sequence_by_dim(x_BLC, 1)

        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):              
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn, rope_cache, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, rope2d_freqs_grid=rope_cache)
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=rope_cache)

        if sp_manager.sp_on():
            # [B, raw_L/sp_size, C] --> [B, raw_L, C]
            x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)

        # [3. unpad the seqlen dim, and then get logits]
        x_BLC = self.pick_visual_tokens(x_BLC, sequece_packing_scales, visual_tokens_len, self.other_args)
        loss_list, acc_list = self.get_loss_acc(x_BLC, sequece_packing_scales, gt_BL)
        return loss_list, acc_list, valid_sequence_ratio

    def prepare_text_conditions(
        self,
        label_B_or_BLT,
        cfg_list,
        B,
        negative_label_B_or_BLT,
        vae_scale_schedule=None,
        text_token_only=False,
        text_maxlen_this_iter=512,
    ):
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        bs = B
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
                lens = lens + lens
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
                lens = lens + lens_un
        kv_compact = self.text_norm(kv_compact)
        kv_compact = self.text_proj(kv_compact).contiguous()
        assert B == 1
        prefix_tokens = torch.zeros((bs, text_maxlen_this_iter, self.C), dtype=kv_compact.dtype, device=kv_compact.device)
        total = 0
        for i, le in enumerate(lens):
            assert le <= text_maxlen_this_iter
            prefix_tokens[i,:le] = kv_compact[total:total+le]
            total += le
        return prefix_tokens, lens
    
    @torch.no_grad()
    def autoregressive_infer(
        self,
        args=None,
        **kwargs,
    ):
        if 'infinity_elegant' in args.dynamic_scale_schedule:
            infer_func = self.ar_infer_infinity_elegant
        elif 'infinity_star_interact' in args.dynamic_scale_schedule:
            infer_func = self.ar_infer_infinity_star_interact
        else:
            infer_func = self.autoregressive_infer_cfg
        return infer_func(args=args, **kwargs)

    def embeds_codes2input(
        self,
        last_stage, # [B, d, t, h, w]
        repeat=1,
    ):
        if self.apply_spatial_patchify: # patchify operation
            last_stage = last_stage.permute(0,2,1,3,4) # [B, t, d, 2h, 2w]
            last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, t, 4d, h, w]
            last_stage = last_stage.permute(0,2,1,3,4) # [B, 4d, t, h, w]
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, t*h*w] or [B, 4d, t*h*w]
        last_stage = torch.permute(last_stage, [0,2,1]) # [B, t*h*w, d] or [B, t*h*w, 4d]
        last_stage = self.word_embed(self.norm0_ve(last_stage))
        last_stage = last_stage.repeat(repeat, 1, 1)
        return last_stage
    
    @torch.no_grad()
    def ar_infer_infinity_elegant(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        context_info=None,
        return_summed_code_only=False,
        kv_pool_keep_ratio=1.0,
        kv_pool_pool_kernel=2,
        kv_pool_threshold=20000,
        prune_keep_ratio=1.0,
        prune_token_threshold=5000,
        prune_ratios_per_scale=None,
        prune_last_iter_full=False,
        stagevar_config=None,
        **kwargs,
    ):   # returns List[idx_Bl]
        from infinity.schedules.infinity_elegant import interpolate
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)

        # Configure KV pooling in attention blocks
        for b in self.unregistered_blocks:
            b.attn.kv_pool_keep_ratio = kv_pool_keep_ratio
            b.attn.kv_pool_pool_kernel = kv_pool_pool_kernel
            b.attn.kv_pool_token_threshold = kv_pool_threshold
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_cfg + args.use_apg == 1
        device = label_B_or_BLT[0].device
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = label_B_or_BLT[-1] # self.text_maxlen # kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        prefix_tokens, lens = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT, vae_scale_schedule, text_token_only=False, text_maxlen_this_iter=text_maxlen_this_iter)
        bs = prefix_tokens.shape[0]
        self._text_emb_for_pruning = prefix_tokens[:B].detach()
        self._text_lens_for_pruning = tuple(int(v) for v in lens[:B])
        ca_kv, cond_BD_or_gss, attn_mask = None, None, None
        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))
        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
        assert len(image_scale_repetition) == scales_in_one_clip, f'{len(image_scale_repetition)} != {scales_in_one_clip}'
        total_steps = image_scale_repetition.sum() + video_scale_repetition.sum() * (len(scale_schedule)//len(video_scale_repetition)-1) + 1 # +1 is prefix text token forward step
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks
        if stagevar_config:
            self._stagevar_output_cache = {}
        self._prune_pu_cache = {}
        self._faststar_scale_cache = {}
        self._fastvar_scale_cache = {}
        self._sparsevar_mse_cache = {}
        self._sparsevar_pn_cache = {}
        self._prune_count_cache = {}
        self._faststar_clip_cache = {}
        self._attn_clip_cache = {}
        self._attn_imp_last = {}
        self._attn_imp_clip_cache = {}

        noise_shape = vae_scale_schedule[0]
        if self.other_args.noise_input:
            noise = torch.randn((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)
        else:
            noise = torch.zeros((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)
        
        summed_codes = [noise[0:1]]
        sos_token = self.embeds_codes2input(noise, bs//1)
        # text tokens forward
        rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter]
        last_stage = prefix_tokens
        pbar.update(1)
        for block_idx, b in enumerate(block_chunks):
            last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind='t0', context_info=context_info, last_repetition_step=True)
        
        # visual tokens forward
        ref_text_scale_inds = ['t0']
        last_stage = sos_token
        cum_scales = 0
        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            rel_si_in_one_clip = si % scales_in_one_clip
            if si < scales_in_one_clip: # image
                repeat_times = image_scale_repetition[si%scales_in_one_clip]
                target_pn = vae_scale_schedule[first_full_spatial_size_scale_index]
            else:
                repeat_times = video_scale_repetition[si%scales_in_one_clip]
                target_pn = vae_scale_schedule[-1]
            cfg = cfg_list[si]
            _adaptive_cfg = getattr(args, 'adaptive_cfg_bypass', False)
            if _adaptive_cfg and cfg != 1 and bs > B:
                _cfg_thr = getattr(args, 'adaptive_cfg_threshold', 0.1)
                with torch.no_grad():
                    _div = (last_stage[:B] - last_stage[B:]).pow(2).mean().sqrt().item()
                if _div < _cfg_thr:
                    if not hasattr(self, '_adaptive_cfg_log'):
                        self._adaptive_cfg_log = []
                    self._adaptive_cfg_log.append(si)
                    cfg = 1
            infer_repeat_times = min(repeat_times, args.max_repeat_times)
            for repeat_idx in range(infer_repeat_times):
                # print(f'real scale ind is : {cum_scales+repeat_idx}')
                rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule, si, cum_scales+repeat_idx, device, args, context_info, first_full_spatial_size_scale_index)
                pbar.update(1)
                last_repetition_step = (repeat_idx == (infer_repeat_times-1))
                _faststar_zero_fill = False
                _use_soft_zf = False
                _zf_topk_hw = None
                _zf_pn = None
                _tome_info = None
                _sparsevar_info = None

                _stagevar_active = stagevar_config and si in stagevar_config
                if _stagevar_active and bs > B:
                    from baselines.stagevar import stagevar_compute_indices, stagevar_gather, stagevar_scatter, stagevar_gather_rope
                    _sv_keep = stagevar_config[si]

                    # α=0 → skip scale entirely (just reuse input as output)
                    if _sv_keep <= 0.0:
                        logits_BlV = self.get_logits_during_infer(last_stage[:B], is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(1/tau_list[si])
                    else:
                        # CFG=0: only forward unconditional half
                        last_stage_uncond = last_stage[B:]
                        _do_select = _sv_keep < 1.0 and pn[1] * pn[2] >= prune_token_threshold
                        if prune_last_iter_full and last_repetition_step:
                            _do_select = False

                        if _do_select:
                            _topk_hw, _flat_idx = stagevar_compute_indices(last_stage_uncond, pn, _sv_keep)
                            _rope_reduced = stagevar_gather_rope(rope_cache, _flat_idx)

                            # Block-level RP+RTR: each block operates on reduced tokens,
                            # then restores full before feeding to next block
                            h_sv = stagevar_gather(last_stage_uncond, _flat_idx)
                            for block_idx, b in enumerate(block_chunks):
                                h_sv = b(x=h_sv, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=_rope_reduced, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds, uncond_only=True)
                            # RTR: scatter block output into cached previous-scale output
                            if not hasattr(self, '_stagevar_output_cache') or si not in self._stagevar_output_cache:
                                _cache_src = last_stage_uncond
                            else:
                                _cache_src = self._stagevar_output_cache[si]
                            last_stage_uncond = stagevar_scatter(h_sv, _flat_idx, _cache_src)
                        else:
                            for block_idx, b in enumerate(block_chunks):
                                last_stage_uncond = b(x=last_stage_uncond, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds, uncond_only=True)

                        # Cache this scale's output for next scale's RTR
                        if not hasattr(self, '_stagevar_output_cache'):
                            self._stagevar_output_cache = {}
                        if last_repetition_step and si + 1 < len(scale_schedule):
                            self._stagevar_output_cache[si + 1] = last_stage_uncond.detach()

                        last_stage = last_stage_uncond.repeat(bs // B, 1, 1)
                        logits_BlV = self.get_logits_during_infer(last_stage_uncond, is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(1/tau_list[si])
                elif cfg == 1 and bs > B:
                    # CFG bypass: only forward conditional half through blocks
                    last_stage_cond = last_stage[:B]
                    _cur_keep = prune_ratios_per_scale.get(si, 1.0) if prune_ratios_per_scale else prune_keep_ratio
                    _do_prune = _cur_keep < 1.0 and pn[1] * pn[2] >= prune_token_threshold
                    _scoring = getattr(args, 'prune_scoring', 'v1')
                    if prune_last_iter_full and last_repetition_step and infer_repeat_times > 1:
                        _do_prune = False

                    # Hybrid pruning: how many leading chunks run full tokens
                    _prune_full_chunks = 0
                    if _do_prune:
                        _pfcs = getattr(args, 'prune_full_chunks_per_scale', None)
                        if _pfcs and si in _pfcs:
                            _prune_full_chunks = _pfcs[si]

                    _sel_mlp = bool(getattr(self.other_args, 'prune_selective_mlp', False) and _do_prune)
                    if os.environ.get("PRUNE_DEBUG") == "1" and not getattr(self, '_dbgdone', False):
                        print(f'[SI] si={si} rep={repeat_idx} dp={_do_prune} sm={getattr(self.other_args, "prune_selective_mlp", "MISS")} ap={getattr(args, "prune_selective_mlp", "MISS")}', flush=True)
                        if si >= 29: self._dbgdone = True
                    if _sel_mlp:
                        from infinity.models.st_token_pruning import prune_tokens
                        _full_cond = last_stage_cond
                        _, _, _flat_idx, _topk_hw = prune_tokens(
                            last_stage_cond, rope_cache, pn, _cur_keep,
                            w_str=getattr(args, 'prune_w_str', 0.5),
                            power_iter=getattr(args, 'prune_power_iter', 3),
                            w_temp=getattr(args, 'prune_w_temp', 1.0),
                            fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                            per_frame=getattr(args, 'prune_per_frame', False),
                            scoring=_scoring,
                            w_act=getattr(args, 'prune_w_act', 0.5),
                            w_conv=getattr(args, 'prune_w_conv', 1.0),
                        )
                        rope_cache_pruned = rope_cache
                        for block_idx, b in enumerate(block_chunks):
                            last_stage_cond = b(x=last_stage_cond, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds, cond_only=True, mlp_keep_idx=_flat_idx)
                        _faststar_zero_fill = True
                        _zf_topk_hw = _topk_hw
                        _zf_pn = pn
                    elif _do_prune and _prune_full_chunks <= 0:
                        # Original path: prune before all blocks
                        if _scoring == 'faststar':
                            from baselines.faststar import faststar_prune_tokens, faststar_partial_update
                            _full_cond = last_stage_cond
                            _prev_scale = self._faststar_scale_cache.get(('bypass', si - 1), self._faststar_scale_cache.get(('normal', si - 1), None))
                            _prev_clip_rel = rel_si_in_one_clip
                            _prev_clip = self._faststar_clip_cache.get(('bypass', _prev_clip_rel), None)
                            last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = faststar_prune_tokens(
                                last_stage_cond, rope_cache, pn, _cur_keep,
                                prev_scale_feat=_prev_scale,
                                prev_clip_feat=_prev_clip,
                                prev_scale_pn=(scale_schedule[si - 1] if si > 0 else None),
                                prev_clip_pn=scale_schedule[rel_si_in_one_clip],
                                fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                            )
                            _faststar_zero_fill = True
                            _zf_topk_hw = _topk_hw
                            _zf_pn = pn
                        elif _scoring == 'fastvar':
                            from baselines.fastvar import fastvar_prune_tokens
                            _full_cond = last_stage_cond
                            _fastvar_prev_cache = self._fastvar_scale_cache.get(('bypass', si - 1), self._fastvar_scale_cache.get(('normal', si - 1), None))
                            _fastvar_prev_pn = scale_schedule[si - 1] if si > 0 else None
                            last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = fastvar_prune_tokens(
                                last_stage_cond, rope_cache, pn, _cur_keep,
                            )
                        elif _scoring == 'sparsevar':
                            from baselines.sparsevar import sparsevar_select_tokens
                            _full_cond = last_stage_cond
                            _prev_key = ('bypass', si - 1)
                            _prev_alt = ('normal', si - 1)
                            _prev_mse = self._sparsevar_mse_cache.get(_prev_key, self._sparsevar_mse_cache.get(_prev_alt, None))
                            _prev_pn = self._sparsevar_pn_cache.get(_prev_key, self._sparsevar_pn_cache.get(_prev_alt, None))
                            last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw, _sparsevar_info = sparsevar_select_tokens(
                                last_stage_cond, rope_cache, pn, _cur_keep,
                                prev_mse=_prev_mse,
                                prev_pn=_prev_pn,
                                anchor_stride=getattr(args, 'sparsevar_anchor_stride', 4),
                            )
                        elif _scoring == 'tome':
                            from baselines.tome import tome_merge_tokens
                            _full_cond = last_stage_cond
                            last_stage_cond, rope_cache_pruned, _flat_idx, _tome_info = tome_merge_tokens(
                                last_stage_cond, rope_cache, pn, _cur_keep,
                                chunk_size=getattr(args, 'tome_chunk_size', 1024),
                            )
                            _topk_hw = None
                        elif _scoring == 'attn':
                            from infinity.models.st_token_pruning import compute_attention_importance, select_attention_tokens
                            _full_cond = last_stage_cond
                            _att = None
                            for _bm in self.modules():
                                if _bm.__class__.__name__ == 'SelfAttention':
                                    _att = _bm; break
                            _text_k = _att.cached_k[0] if (isinstance(_att.cached_k, list) and len(_att.cached_k) > 0) else _att.cached_k.get('t0', _att.cached_k.get(0, None))
                            _prev_clip_h = self._attn_clip_cache.get(('bypass', rel_si_in_one_clip), None)
                            _imp = compute_attention_importance(
                                last_stage_cond, pn, _att,
                                text_k=_text_k,
                                text_lens=self._text_lens_for_pruning if getattr(args, 'prune_attn_mask_text_padding', False) else None,
                                prev_clip_hidden=_prev_clip_h,
                                w_spatial=getattr(args, 'prune_w_spatial', 1.0),
                                w_temporal=getattr(args, 'prune_w_temporal', 1.0),
                                w_text=getattr(args, 'prune_w_text', 1.0),
                                fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                                query_stride=getattr(args, 'prune_attn_query_stride', None),
                                max_query_tokens=getattr(args, 'prune_attn_max_queries', None),
                                rope_cache=rope_cache,
                                per_frame=getattr(args, 'prune_per_frame', False),
                                text_pool=getattr(args, 'prune_attn_text_pool', 'max'),
                                text_topk=getattr(args, 'prune_attn_text_topk', 4),
                                temporal_smooth=getattr(args, 'prune_attn_temporal_smooth', 0.0),
                                text_query_gate=getattr(args, 'prune_attn_text_query_gate', 0.0),
                                temporal_mode=getattr(args, 'prune_attn_temporal_mode', 'column'),
                                prev_importance=self._attn_imp_clip_cache.get(rel_si_in_one_clip),
                                temporal_anchors=getattr(args, 'prune_attn_temporal_anchors', 1024),
                                random_select=getattr(args, 'prune_attn_random_select', False),
                            )
                            self._attn_imp_last[rel_si_in_one_clip] = _imp.detach()
                            last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = select_attention_tokens(
                                last_stage_cond, rope_cache, pn, _imp, _cur_keep,
                                shared_fraction=getattr(args, 'prune_attn_shared_fraction', 0.0),
                            )
                            _faststar_zero_fill = True
                            _zf_topk_hw = _topk_hw
                            _zf_pn = pn
                        elif _scoring == 'stepvar':
                            from baselines.stepvar import stepvar_prune_tokens
                            _full_cond = last_stage_cond
                            last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = stepvar_prune_tokens(
                                last_stage_cond, rope_cache, pn, _cur_keep,
                                w_str=getattr(args, 'prune_w_str', 0.5),
                                power_iter=getattr(args, 'prune_power_iter', 3),
                            )
                        else:
                            from infinity.models.st_token_pruning import prune_tokens, partial_update_recover
                            _full_cond = last_stage_cond
                            _pu_prev = self._prune_pu_cache.get(('bypass', si), None) if _scoring == 'v2' else None
                            last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = prune_tokens(
                                last_stage_cond, rope_cache, pn, _cur_keep,
                                w_str=getattr(args, 'prune_w_str', 0.5),
                                power_iter=getattr(args, 'prune_power_iter', 3),
                                w_temp=getattr(args, 'prune_w_temp', 1.0),
                                fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                                per_frame=getattr(args, 'prune_per_frame', False),
                                scoring=_scoring, pu_cache=_pu_prev,
                                w_act=getattr(args, 'prune_w_act', 0.5),
                                w_conv=getattr(args, 'prune_w_conv', 1.0),
                                text_emb=self._text_emb_for_pruning if getattr(args, 'prune_scoring', 'v1') == 'v3' else None,
                                prev_scale_feat=self._faststar_scale_cache.get(('bypass', si - 1), self._faststar_scale_cache.get(('normal', si - 1), None)) if _scoring == 'v4' else None,
                                prune_counts=self._prune_count_cache.get(('bypass', si - 1), None) if getattr(args, 'prune_error_aware', False) else None,
                            )
                        if getattr(args, "prune_zero_fill", False):
                            _faststar_zero_fill = True
                            _zf_topk_hw = _topk_hw
                            _zf_pn = pn
                        if getattr(args, 'prune_mode', 'prune') == 'merge':
                            from infinity.models.st_token_pruning import enrich_selected_tokens
                            last_stage_cond = enrich_selected_tokens(last_stage_cond, _full_cond, _topk_hw, pn, alpha=getattr(args, 'merge_alpha', 0.3))
                        for block_idx, b in enumerate(block_chunks):
                            last_stage_cond = b(x=last_stage_cond, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache_pruned, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds, cond_only=True)
                    elif _do_prune and _prune_full_chunks > 0:
                        # Hybrid path: first N chunks full tokens, then prune for remaining chunks
                        _full_cond = last_stage_cond
                        for block_idx, b in enumerate(block_chunks):
                            if block_idx == _prune_full_chunks:
                                if _scoring == 'faststar':
                                    from baselines.faststar import faststar_prune_tokens, faststar_partial_update
                                    _prev_scale = self._faststar_scale_cache.get(('bypass', si - 1), self._faststar_scale_cache.get(('normal', si - 1), None))
                                    _prev_clip_rel = rel_si_in_one_clip
                                    _prev_clip = self._faststar_clip_cache.get(('bypass', _prev_clip_rel), None)
                                    last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = faststar_prune_tokens(
                                        last_stage_cond, rope_cache, pn, _cur_keep,
                                        prev_scale_feat=_prev_scale,
                                        prev_clip_feat=_prev_clip,
                                        prev_scale_pn=(scale_schedule[si - 1] if si > 0 else None),
                                        prev_clip_pn=scale_schedule[rel_si_in_one_clip],
                                        fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                                    )
                                    _faststar_zero_fill = True
                                    _zf_topk_hw = _topk_hw
                                    _zf_pn = pn
                                elif _scoring == 'fastvar':
                                    from baselines.fastvar import fastvar_prune_tokens
                                    _fastvar_prev_cache = self._fastvar_scale_cache.get(('bypass', si - 1), self._fastvar_scale_cache.get(('normal', si - 1), None))
                                    _fastvar_prev_pn = scale_schedule[si - 1] if si > 0 else None
                                    last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = fastvar_prune_tokens(
                                        last_stage_cond, rope_cache, pn, _cur_keep,
                                    )
                                elif _scoring == 'sparsevar':
                                    from baselines.sparsevar import sparsevar_select_tokens
                                    _prev_key = ('bypass', si - 1)
                                    _prev_alt = ('normal', si - 1)
                                    _prev_mse = self._sparsevar_mse_cache.get(_prev_key, self._sparsevar_mse_cache.get(_prev_alt, None))
                                    _prev_pn = self._sparsevar_pn_cache.get(_prev_key, self._sparsevar_pn_cache.get(_prev_alt, None))
                                    last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw, _sparsevar_info = sparsevar_select_tokens(
                                        last_stage_cond, rope_cache, pn, _cur_keep,
                                        prev_mse=_prev_mse,
                                        prev_pn=_prev_pn,
                                        anchor_stride=getattr(args, 'sparsevar_anchor_stride', 4),
                                    )
                                elif _scoring == 'tome':
                                    from baselines.tome import tome_merge_tokens
                                    last_stage_cond, rope_cache_pruned, _flat_idx, _tome_info = tome_merge_tokens(
                                        last_stage_cond, rope_cache, pn, _cur_keep,
                                        chunk_size=getattr(args, 'tome_chunk_size', 1024),
                                    )
                                    _topk_hw = None
                                elif _scoring == 'attn':
                                    from infinity.models.st_token_pruning import compute_attention_importance, select_attention_tokens
                                    _att = None
                                    for _bm in self.modules():
                                        if _bm.__class__.__name__ == 'SelfAttention':
                                            _att = _bm; break
                                    _text_k = _att.cached_k[0] if (isinstance(_att.cached_k, list) and len(_att.cached_k) > 0) else _att.cached_k.get('t0', _att.cached_k.get(0, None))
                                    _prev_clip_h = self._attn_clip_cache.get(('bypass', rel_si_in_one_clip), None)
                                    _imp = compute_attention_importance(
                                        last_stage_cond, pn, _att,
                                        text_k=_text_k,
                                        text_lens=self._text_lens_for_pruning if getattr(args, 'prune_attn_mask_text_padding', False) else None,
                                        prev_clip_hidden=_prev_clip_h,
                                        w_spatial=getattr(args, 'prune_w_spatial', 1.0),
                                        w_temporal=getattr(args, 'prune_w_temporal', 1.0),
                                        w_text=getattr(args, 'prune_w_text', 1.0),
                                        fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                                        query_stride=getattr(args, 'prune_attn_query_stride', None),
                                        max_query_tokens=getattr(args, 'prune_attn_max_queries', None),
                                        rope_cache=rope_cache,
                                        per_frame=getattr(args, 'prune_per_frame', False),
                                        text_pool=getattr(args, 'prune_attn_text_pool', 'max'),
                                        text_topk=getattr(args, 'prune_attn_text_topk', 4),
                                        temporal_smooth=getattr(args, 'prune_attn_temporal_smooth', 0.0),
                                        text_query_gate=getattr(args, 'prune_attn_text_query_gate', 0.0),
                                        temporal_mode=getattr(args, 'prune_attn_temporal_mode', 'column'),
                                        prev_importance=self._attn_imp_clip_cache.get(rel_si_in_one_clip),
                                        temporal_anchors=getattr(args, 'prune_attn_temporal_anchors', 1024),
                                        random_select=getattr(args, 'prune_attn_random_select', False),
                                    )
                                    self._attn_imp_last[rel_si_in_one_clip] = _imp.detach()
                                    last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = select_attention_tokens(
                                        last_stage_cond, rope_cache, pn, _imp, _cur_keep,
                                        shared_fraction=getattr(args, 'prune_attn_shared_fraction', 0.0),
                                    )
                                    _faststar_zero_fill = True
                                    _zf_topk_hw = _topk_hw
                                    _zf_pn = pn
                                elif _scoring == 'stepvar':
                                    from baselines.stepvar import stepvar_prune_tokens
                                    last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = stepvar_prune_tokens(
                                        last_stage_cond, rope_cache, pn, _cur_keep,
                                        w_str=getattr(args, 'prune_w_str', 0.5),
                                        power_iter=getattr(args, 'prune_power_iter', 3),
                                    )
                                else:
                                    from infinity.models.st_token_pruning import prune_tokens, partial_update_recover
                                    _pu_prev = self._prune_pu_cache.get(('bypass', si), None) if _scoring == 'v2' else None
                                    last_stage_cond, rope_cache_pruned, _flat_idx, _topk_hw = prune_tokens(
                                        last_stage_cond, rope_cache, pn, _cur_keep,
                                        w_str=getattr(args, 'prune_w_str', 0.5),
                                        power_iter=getattr(args, 'prune_power_iter', 3),
                                        w_temp=getattr(args, 'prune_w_temp', 1.0),
                                        fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                                        per_frame=getattr(args, 'prune_per_frame', False),
                                        scoring=_scoring, pu_cache=_pu_prev,
                                        w_act=getattr(args, 'prune_w_act', 0.5),
                                        w_conv=getattr(args, 'prune_w_conv', 1.0),
                                        text_emb=self._text_emb_for_pruning if getattr(args, 'prune_scoring', 'v1') == 'v3' else None,
                                        prev_scale_feat=self._faststar_scale_cache.get(('bypass', si - 1), self._faststar_scale_cache.get(('normal', si - 1), None)) if _scoring == 'v4' else None,
                                    )
                                if getattr(args, "prune_zero_fill", False):
                                    _faststar_zero_fill = True
                                    _zf_topk_hw = _topk_hw
                                    _zf_pn = pn
                            _cur_rope = rope_cache if block_idx < _prune_full_chunks else rope_cache_pruned
                            last_stage_cond = b(x=last_stage_cond, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=_cur_rope, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds, cond_only=True)
                    else:
                        rope_cache_pruned = rope_cache
                        for block_idx, b in enumerate(block_chunks):
                            last_stage_cond = b(x=last_stage_cond, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds, cond_only=True)
                    if _do_prune and not _sel_mlp:
                        if _scoring == 'faststar':
                            _pu_base = self._prune_pu_cache.get(('bypass', si), _full_cond)
                            last_stage_cond = faststar_partial_update(last_stage_cond, _flat_idx, _pu_base)
                        elif _scoring == 'fastvar':
                            from baselines.fastvar import fastvar_cached_recover
                            last_stage_cond = fastvar_cached_recover(
                                last_stage_cond, _flat_idx, _full_cond,
                                prev_scale_cache=_fastvar_prev_cache,
                                prev_pn=_fastvar_prev_pn,
                                pn=pn,
                            )
                        elif _scoring == 'sparsevar':
                            from baselines.sparsevar import sparsevar_anchor_recover, sparsevar_change_map
                            last_stage_cond = sparsevar_anchor_recover(last_stage_cond, _sparsevar_info, _full_cond)
                            self._sparsevar_mse_cache[('bypass', si)] = sparsevar_change_map(last_stage_cond, _full_cond, pn).detach()
                            self._sparsevar_pn_cache[('bypass', si)] = pn
                        elif _scoring == 'tome':
                            from baselines.tome import tome_unmerge_tokens
                            last_stage_cond = tome_unmerge_tokens(last_stage_cond, _tome_info)
                        elif _scoring == 'attn':
                            from infinity.models.st_token_pruning import recover_attention_tokens
                            _pu_base = self._prune_pu_cache.get(('bypass', si), _full_cond)
                            _recover_mode = getattr(args, 'prune_recover_mode', 'pu')
                            last_stage_cond = recover_attention_tokens(
                                last_stage_cond, _flat_idx, _pu_base, _full_cond, pn, _topk_hw,
                                per_frame=getattr(args, 'prune_per_frame', False),
                                mode=_recover_mode,
                                n_neighbors=getattr(args, 'prune_recover_neighbors', 4),
                                residual_scale=getattr(args, 'prune_recover_alpha', 1.0),
                            )
                            _code_mask = getattr(args, 'prune_recover_code_mask', None)
                            _faststar_zero_fill = (
                                _recover_mode == 'pu' if _code_mask is None else bool(_code_mask)
                            )
                        elif _scoring == 'stepvar':
                            from baselines.stepvar import stepvar_nn_recover
                            last_stage_cond = stepvar_nn_recover(last_stage_cond, _topk_hw, pn, last_stage_cond.device)
                            self._prune_pu_cache[('bypass', si)] = last_stage_cond.detach()
                        else:
                            _pu_base = self._prune_pu_cache.get(('bypass', si), _full_cond)
                            _recover_mode = getattr(args, 'prune_recover_mode', 'pu')
                            if _recover_mode == 'delta_prop':
                                from infinity.models.st_token_pruning import delta_propagation_recover
                                last_stage_cond = delta_propagation_recover(last_stage_cond, _flat_idx, _full_cond, pn, _topk_hw, per_frame=getattr(args, 'prune_per_frame', False))
                                _faststar_zero_fill = True
                            elif _recover_mode == 'interp':
                                from infinity.models.st_token_pruning import partial_update_recover_interp
                                last_stage_cond = partial_update_recover_interp(last_stage_cond, _flat_idx, _pu_base, pn, _topk_hw, per_frame=getattr(args, 'prune_per_frame', False))
                                _faststar_zero_fill = False
                            elif _recover_mode == 'nn':
                                from infinity.models.st_token_pruning import recover_tokens
                                last_stage_cond = recover_tokens(last_stage_cond, _topk_hw, pn, last_stage_cond.device, per_frame=getattr(args, 'prune_per_frame', False))
                                _faststar_zero_fill = False
                            else:
                                if getattr(args, 'prune_mode', 'prune') == 'merge':
                                    from infinity.models.st_token_pruning import unmerge_recover
                                    last_stage_cond = unmerge_recover(last_stage_cond, _topk_hw, pn, pu_base=_pu_base, beta=getattr(args, 'merge_beta', 0.3))
                                else:
                                    last_stage_cond = partial_update_recover(last_stage_cond, _flat_idx, _pu_base)
                        self._prune_pu_cache[('bypass', si)] = last_stage_cond.detach()
                    if getattr(args, 'prune_error_aware', False) and _do_prune:
                        _prev_counts = self._prune_count_cache.get(('bypass', si), torch.zeros(1, pn[1]*pn[2], device=last_stage_cond.device))
                        _all_hw = torch.arange(pn[1]*pn[2], device=last_stage_cond.device).unsqueeze(0).expand(1, -1)
                        _pruned_mask = torch.ones(1, pn[1]*pn[2], device=last_stage_cond.device)
                        _pruned_mask.scatter_(1, _topk_hw[:1], 0.0)
                        self._prune_count_cache[('bypass', si)] = _prev_counts + _pruned_mask
                    # FastSTAR: always cache features for next scale/clip scoring
                    if _scoring == 'faststar':
                        self._faststar_scale_cache[('bypass', si)] = last_stage_cond.detach()
                        if last_repetition_step:
                            self._faststar_clip_cache[('bypass', rel_si_in_one_clip)] = last_stage_cond.detach()
                    elif _scoring == 'fastvar':
                        self._fastvar_scale_cache[('bypass', si)] = last_stage_cond.detach()
                    elif _scoring == 'v4':
                        self._faststar_scale_cache[('bypass', si)] = last_stage_cond.detach()
                    elif _scoring == 'attn':
                        if last_repetition_step:
                            self._attn_clip_cache[('bypass', rel_si_in_one_clip)] = last_stage_cond.detach()
                            if rel_si_in_one_clip in self._attn_imp_last:
                                self._attn_imp_clip_cache[rel_si_in_one_clip] = self._attn_imp_last[rel_si_in_one_clip]
                    last_stage = last_stage_cond.repeat(bs // B, 1, 1)
                    logits_BlV = self.get_logits_during_infer(last_stage_cond, is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(1/tau_list[si])
                else:
                    _cur_keep = prune_ratios_per_scale.get(si, 1.0) if prune_ratios_per_scale else prune_keep_ratio
                    _do_prune = _cur_keep < 1.0 and pn[1] * pn[2] >= prune_token_threshold
                    _scoring = getattr(args, 'prune_scoring', 'v1')
                    if prune_last_iter_full and last_repetition_step and infer_repeat_times > 1:
                        _do_prune = False
                    _ctx_n = 0
                    if _do_prune:
                        _full_stage = last_stage.clone()
                        cond_half = last_stage[:B]
                        uncond_half = last_stage[B:]
                        if _scoring == 'v32':
                            # cross-attention proxy: visual q (model's projection) vs cached TEXT keys (approx, no rope on q)
                            import torch.nn.functional as _F32
                            _att = None
                            for _bm in self.modules():
                                if _bm.__class__.__name__ == 'SelfAttention':
                                    _att = _bm; break
                            _L32 = cond_half.shape[1]
                            # qwen arch: q_proj + q_norm, scale=head_dim**-0.5
                            _q32 = _att.q_proj(cond_half).view(B, _L32, _att.num_heads, _att.head_dim)
                            _q32 = _att.q_norm(_q32).permute(0, 2, 1, 3).float()  # [B, H, L_v, d]
                            _kt = _att.cached_k[0].float()  # [B, H_kv, L_text, d]
                            # repeat KV heads if grouped-query attention
                            if _kt.shape[1] != _q32.shape[1]:
                                _rep = _q32.shape[1] // _kt.shape[1]
                                _kt = _kt.repeat_interleave(_rep, dim=1)
                            if _kt.shape[0] != _q32.shape[0]:
                                _kt = _kt[:_q32.shape[0]]
                            _scale32 = _att.head_dim ** -0.5
                            _aw = torch.matmul(_q32, _kt.transpose(-1, -2)) * _scale32
                            _tattn = _aw.amax(dim=-1).mean(dim=1)
                            _ta = _tattn.view(B, pn[0], pn[1] * pn[2]).mean(dim=1)
                            _xf = cond_half.float().view(B, pn[0], pn[1] * pn[2], -1)
                            _tv = _xf.var(dim=1).sum(dim=-1) if pn[0] >= 2 else torch.zeros_like(_ta)
                            def _n01(s):
                                mn = s.min(dim=-1, keepdim=True).values; mx = s.max(dim=-1, keepdim=True).values; return (s - mn) / (mx - mn + 1e-8)
                            _imp = (_n01(_ta).pow(2) + _n01(_tv).pow(2)).pow(0.5)
                            _ok = max(1, int(pn[1] * pn[2] * _cur_keep))
                            _, _topk_hw = _imp.topk(_ok, dim=-1)
                            _topk_hw, _ = _topk_hw.sort(dim=-1)
                            _ofo = torch.arange(pn[0], device=cond_half.device).unsqueeze(1) * (pn[1] * pn[2])
                            _flat_idx = (_topk_hw.unsqueeze(1) + _ofo.unsqueeze(0)).reshape(B, pn[0] * _ok)
                            cond_half = cond_half.gather(1, _flat_idx.unsqueeze(-1).expand(-1, -1, cond_half.shape[-1]))
                            rope_cache_pruned = rope_cache[:, :, :, :, _flat_idx[0], :]
                            _faststar_zero_fill = True
                            _zf_topk_hw = _topk_hw
                            _zf_pn = pn
                        elif _scoring == 'oracle':
                            # CEILING PROBE: full forward of cond to get TRUE per-token change, keep top-k by it
                            _probe = cond_half
                            for _ob in block_chunks:
                                _probe = _ob(x=_probe, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds, cond_only=True)
                            _chg = (_probe.float() - cond_half.float()).view(B, pn[0], pn[1] * pn[2], -1).norm(dim=-1).mean(dim=1)
                            del _probe; torch.cuda.empty_cache()
                            _ok = max(1, int(pn[1] * pn[2] * _cur_keep))
                            _, _topk_hw = _chg.topk(_ok, dim=-1)
                            _topk_hw, _ = _topk_hw.sort(dim=-1)
                            _ofo = torch.arange(pn[0], device=cond_half.device).unsqueeze(1) * (pn[1] * pn[2])
                            _flat_idx = (_topk_hw.unsqueeze(1) + _ofo.unsqueeze(0)).reshape(B, pn[0] * _ok)
                            cond_half = cond_half.gather(1, _flat_idx.unsqueeze(-1).expand(-1, -1, cond_half.shape[-1]))
                            rope_cache_pruned = rope_cache[:, :, :, :, _flat_idx[0], :]
                            _faststar_zero_fill = True
                            _zf_topk_hw = _topk_hw
                            _zf_pn = pn
                        elif _scoring == 'attn':
                            from infinity.models.st_token_pruning import compute_attention_importance, select_attention_tokens
                            _att = None
                            for _bm in self.modules():
                                if _bm.__class__.__name__ == 'SelfAttention':
                                    _att = _bm; break
                            _text_k = _att.cached_k[0] if (isinstance(_att.cached_k, list) and len(_att.cached_k) > 0) else _att.cached_k.get('t0', _att.cached_k.get(0, None))
                            _prev_clip_h = self._attn_clip_cache.get(('normal', rel_si_in_one_clip), self._attn_clip_cache.get(('bypass', rel_si_in_one_clip), None))
                            _imp = compute_attention_importance(
                                cond_half, pn, _att,
                                text_k=_text_k,
                                text_lens=self._text_lens_for_pruning if getattr(args, 'prune_attn_mask_text_padding', False) else None,
                                prev_clip_hidden=_prev_clip_h,
                                w_spatial=getattr(args, 'prune_w_spatial', 1.0),
                                w_temporal=getattr(args, 'prune_w_temporal', 1.0),
                                w_text=getattr(args, 'prune_w_text', 1.0),
                                fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                                query_stride=getattr(args, 'prune_attn_query_stride', None),
                                max_query_tokens=getattr(args, 'prune_attn_max_queries', None),
                                rope_cache=rope_cache,
                                per_frame=getattr(args, 'prune_per_frame', False),
                                text_pool=getattr(args, 'prune_attn_text_pool', 'max'),
                                text_topk=getattr(args, 'prune_attn_text_topk', 4),
                                temporal_smooth=getattr(args, 'prune_attn_temporal_smooth', 0.0),
                                text_query_gate=getattr(args, 'prune_attn_text_query_gate', 0.0),
                                temporal_mode=getattr(args, 'prune_attn_temporal_mode', 'column'),
                                prev_importance=self._attn_imp_clip_cache.get(rel_si_in_one_clip),
                                temporal_anchors=getattr(args, 'prune_attn_temporal_anchors', 1024),
                                random_select=getattr(args, 'prune_attn_random_select', False),
                            )
                            self._attn_imp_last[rel_si_in_one_clip] = _imp.detach()
                            _ok = max(1, int(pn[1] * pn[2] * _cur_keep))
                            import os as _os_idp
                            if _os_idp.environ.get("ATTN_IDP") == "1" and _imp.ndim == 2:
                                from infinity.models.st_token_pruning import idpruner_select
                                _feat = cond_half.view(B, pn[0], pn[1]*pn[2], cond_half.shape[-1]).mean(1)
                                _topk_hw = idpruner_select(_imp, _feat, _ok, float(_os_idp.environ.get("ATTN_LAM", "0.5")))
                                _ofo = torch.arange(pn[0], device=cond_half.device).unsqueeze(1) * (pn[1] * pn[2])
                                _flat_idx = (_topk_hw.unsqueeze(1) + _ofo.unsqueeze(0)).reshape(B, pn[0] * _ok)
                                cond_half = cond_half.gather(1, _flat_idx.unsqueeze(-1).expand(-1, -1, cond_half.shape[-1]))
                                rope_cache_pruned = rope_cache[:, :, :, :, _flat_idx[0], :]
                            else:
                                cond_half, rope_cache_pruned, _flat_idx, _topk_hw = select_attention_tokens(
                                    cond_half, rope_cache, pn, _imp, _cur_keep,
                                    shared_fraction=getattr(args, 'prune_attn_shared_fraction', 0.0),
                                )
                            _faststar_zero_fill = True
                            _zf_topk_hw = _topk_hw
                            _zf_pn = pn
                        elif _scoring == 'faststar':
                            from baselines.faststar import faststar_prune_tokens, faststar_partial_update
                            _prev_scale = self._faststar_scale_cache.get(('normal', si - 1), self._faststar_scale_cache.get(('bypass', si - 1), None))
                            _prev_clip = self._faststar_clip_cache.get(('normal', rel_si_in_one_clip), None)
                            cond_half, rope_cache_pruned, _flat_idx, _topk_hw = faststar_prune_tokens(
                                cond_half, rope_cache, pn, _cur_keep,
                                prev_scale_feat=_prev_scale,
                                prev_clip_feat=_prev_clip,
                                prev_scale_pn=(scale_schedule[si - 1] if si > 0 else None),
                                prev_clip_pn=scale_schedule[rel_si_in_one_clip],
                                fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                            )
                            _faststar_zero_fill = True
                            _zf_topk_hw = _topk_hw
                            _zf_pn = pn
                        elif _scoring == 'fastvar':
                            from baselines.fastvar import fastvar_prune_tokens
                            _fastvar_prev_cache = self._fastvar_scale_cache.get(('normal', si - 1), self._fastvar_scale_cache.get(('bypass', si - 1), None))
                            _fastvar_prev_cache_uncond = self._fastvar_scale_cache.get(('normal_uncond', si - 1), _fastvar_prev_cache)
                            _fastvar_prev_pn = scale_schedule[si - 1] if si > 0 else None
                            cond_half, rope_cache_pruned, _flat_idx, _topk_hw = fastvar_prune_tokens(
                                cond_half, rope_cache, pn, _cur_keep,
                            )
                        elif _scoring == 'sparsevar':
                            from baselines.sparsevar import sparsevar_select_tokens
                            _prev_key = ('normal', si - 1)
                            _prev_alt = ('bypass', si - 1)
                            _prev_mse = self._sparsevar_mse_cache.get(_prev_key, self._sparsevar_mse_cache.get(_prev_alt, None))
                            _prev_pn = self._sparsevar_pn_cache.get(_prev_key, self._sparsevar_pn_cache.get(_prev_alt, None))
                            cond_half, rope_cache_pruned, _flat_idx, _topk_hw, _sparsevar_info = sparsevar_select_tokens(
                                cond_half, rope_cache, pn, _cur_keep,
                                prev_mse=_prev_mse,
                                prev_pn=_prev_pn,
                                anchor_stride=getattr(args, 'sparsevar_anchor_stride', 4),
                            )
                        elif _scoring == 'tome':
                            from baselines.tome import tome_merge_tokens
                            cond_half, rope_cache_pruned, _flat_idx, _tome_info = tome_merge_tokens(
                                cond_half, rope_cache, pn, _cur_keep,
                                chunk_size=getattr(args, 'tome_chunk_size', 1024),
                            )
                            _topk_hw = None
                        elif _scoring == 'stepvar':
                            from baselines.stepvar import stepvar_prune_tokens
                            cond_half, rope_cache_pruned, _flat_idx, _topk_hw = stepvar_prune_tokens(
                                cond_half, rope_cache, pn, _cur_keep,
                                w_str=getattr(args, 'prune_w_str', 0.5),
                                power_iter=getattr(args, 'prune_power_iter', 3),
                            )
                        else:
                            from infinity.models.st_token_pruning import prune_tokens, partial_update_recover
                            _pu_prev_cond = self._prune_pu_cache.get(('normal', si), None)
                            _pu_for_scoring = _pu_prev_cond[:B] if (_scoring == 'v2' and _pu_prev_cond is not None) else None
                            cond_half, rope_cache_pruned, _flat_idx, _topk_hw = prune_tokens(
                                cond_half, rope_cache, pn, _cur_keep,
                                w_str=getattr(args, 'prune_w_str', 0.5),
                                power_iter=getattr(args, 'prune_power_iter', 3),
                                w_temp=getattr(args, 'prune_w_temp', 1.0),
                                fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                                per_frame=getattr(args, 'prune_per_frame', False),
                                scoring=_scoring, pu_cache=_pu_for_scoring,
                                w_act=getattr(args, 'prune_w_act', 0.5),
                                w_conv=getattr(args, 'prune_w_conv', 1.0),
                                text_emb=self._text_emb_for_pruning if getattr(args, 'prune_scoring', 'v1') == 'v3' else None,
                                prev_scale_feat=self._faststar_scale_cache.get(('normal', si - 1), self._faststar_scale_cache.get(('bypass', si - 1), None)) if _scoring == 'v4' else None,
                                prune_counts=self._prune_count_cache.get(('normal', si - 1), None) if getattr(args, 'prune_error_aware', False) else None,
                            )
                        if getattr(args, "prune_zero_fill", False):
                            _faststar_zero_fill = True
                            _zf_topk_hw = _topk_hw
                            _zf_pn = pn
                        if _scoring == 'tome':
                            from baselines.tome import tome_apply_merge
                            uncond_half = tome_apply_merge(uncond_half, _tome_info)
                        else:
                            uncond_half = uncond_half.gather(1, _flat_idx.unsqueeze(-1).expand(-1, -1, uncond_half.shape[-1]))
                        if getattr(args, 'prune_mode', 'prune') == 'merge':
                            from infinity.models.st_token_pruning import enrich_selected_tokens
                            cond_half = enrich_selected_tokens(cond_half, _full_stage[:B], _topk_hw, pn, alpha=getattr(args, 'merge_alpha', 0.3))
                            uncond_half = enrich_selected_tokens(uncond_half, _full_stage[B:], _topk_hw, pn, alpha=getattr(args, 'merge_alpha', 0.3))
                        _ctx_pool = int(getattr(args, 'prune_ctx_pool', 0) or 0)
                        if _ctx_pool > 1 and _scoring != 'tome':
                            from infinity.models.st_token_pruning import build_context_tokens
                            _ctx_c, _ctx_u, _ctx_rope = build_context_tokens(
                                _full_stage[:B], _full_stage[B:], rope_cache, pn, _flat_idx, _ctx_pool)
                            cond_half = torch.cat([cond_half, _ctx_c], dim=1)
                            uncond_half = torch.cat([uncond_half, _ctx_u], dim=1)
                            rope_cache_pruned = torch.cat([rope_cache_pruned, _ctx_rope], dim=4)
                            _ctx_n = _ctx_c.shape[1]
                        last_stage = torch.cat([cond_half, uncond_half], dim=0)
                        rope_cache_fwd = rope_cache_pruned
                    else:
                        rope_cache_fwd = rope_cache
                    _sel_mlp = bool(getattr(self.other_args, 'prune_selective_mlp', False) and _do_prune)
                    if _sel_mlp:
                        last_stage = _full_stage
                        _mlp_idx = _flat_idx.repeat(_full_stage.shape[0] // _flat_idx.shape[0], 1)
                        for block_idx, b in enumerate(block_chunks):
                            last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds, mlp_keep_idx=_mlp_idx)
                        _faststar_zero_fill = True
                        _zf_topk_hw = _topk_hw
                        _zf_pn = pn
                        if os.environ.get("PRUNE_DEBUG") == "1" and si == 29 and not getattr(self,'_se_dbg',False):
                            print(f'[SELMLP-ELSE] si={si} fullshape={tuple(_full_stage.shape)} kept={_mlp_idx.shape[1]}', flush=True)
                            self._se_dbg = True
                    else:
                        for block_idx, b in enumerate(block_chunks):
                            last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache_fwd, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds)
                        if _ctx_n:
                            last_stage = last_stage[:, :last_stage.shape[1] - _ctx_n]   # drop context tokens
                    if _do_prune and not _sel_mlp:
                        if _scoring == 'attn':
                            from infinity.models.st_token_pruning import recover_attention_tokens
                            _pu_base = self._prune_pu_cache.get(('normal', si), _full_stage)
                            _recover_mode = getattr(args, 'prune_recover_mode', 'pu')
                            _recover_kwargs = {
                                'per_frame': getattr(args, 'prune_per_frame', False),
                                'mode': _recover_mode,
                                'n_neighbors': getattr(args, 'prune_recover_neighbors', 4),
                                'residual_scale': getattr(args, 'prune_recover_alpha', 1.0),
                            }
                            cond_recovered = recover_attention_tokens(
                                last_stage[:B], _flat_idx, _pu_base[:B],
                                _full_stage[:B], pn, _topk_hw, **_recover_kwargs,
                            )
                            uncond_recovered = recover_attention_tokens(
                                last_stage[B:], _flat_idx, _pu_base[B:],
                                _full_stage[B:], pn, _topk_hw, **_recover_kwargs,
                            )
                            _code_mask = getattr(args, 'prune_recover_code_mask', None)
                            _faststar_zero_fill = (
                                _recover_mode == 'pu' if _code_mask is None else bool(_code_mask)
                            )
                        elif _scoring == 'faststar':
                            _pu_base = self._prune_pu_cache.get(('normal', si), _full_stage)
                            cond_recovered = faststar_partial_update(last_stage[:B], _flat_idx, _pu_base[:B])
                            uncond_recovered = faststar_partial_update(last_stage[B:], _flat_idx, _pu_base[B:])
                        elif _scoring == 'fastvar':
                            from baselines.fastvar import fastvar_cached_recover
                            cond_recovered = fastvar_cached_recover(
                                last_stage[:B], _flat_idx, _full_stage[:B],
                                prev_scale_cache=_fastvar_prev_cache,
                                prev_pn=_fastvar_prev_pn,
                                pn=pn,
                            )
                            uncond_recovered = fastvar_cached_recover(
                                last_stage[B:], _flat_idx, _full_stage[B:],
                                prev_scale_cache=_fastvar_prev_cache_uncond,
                                prev_pn=_fastvar_prev_pn,
                                pn=pn,
                            )
                        elif _scoring == 'sparsevar':
                            from baselines.sparsevar import sparsevar_anchor_recover, sparsevar_change_map
                            cond_recovered = sparsevar_anchor_recover(last_stage[:B], _sparsevar_info, _full_stage[:B])
                            uncond_recovered = sparsevar_anchor_recover(last_stage[B:], _sparsevar_info, _full_stage[B:])
                            self._sparsevar_mse_cache[('normal', si)] = sparsevar_change_map(cond_recovered, _full_stage[:B], pn).detach()
                            self._sparsevar_pn_cache[('normal', si)] = pn
                        elif _scoring == 'tome':
                            from baselines.tome import tome_unmerge_tokens
                            cond_recovered = tome_unmerge_tokens(last_stage[:B], _tome_info)
                            uncond_recovered = tome_unmerge_tokens(last_stage[B:], _tome_info)
                        elif _scoring == 'stepvar':
                            from baselines.stepvar import stepvar_nn_recover
                            cond_recovered = stepvar_nn_recover(last_stage[:B], _topk_hw, pn, last_stage.device)
                            uncond_recovered = stepvar_nn_recover(last_stage[B:], _topk_hw, pn, last_stage.device)
                        else:
                            from infinity.models.st_token_pruning import partial_update_recover
                            _pu_base = self._prune_pu_cache.get(('normal', si), _full_stage)
                            _recover_mode = getattr(args, 'prune_recover_mode', 'pu')
                            if _recover_mode == 'interp':
                                from infinity.models.st_token_pruning import partial_update_recover_interp
                                cond_recovered = partial_update_recover_interp(last_stage[:B], _flat_idx, _pu_base[:B], pn, _topk_hw, per_frame=getattr(args, 'prune_per_frame', False))
                                uncond_recovered = partial_update_recover_interp(last_stage[B:], _flat_idx, _pu_base[B:], pn, _topk_hw, per_frame=getattr(args, 'prune_per_frame', False))
                            elif _recover_mode == 'nn':
                                from infinity.models.st_token_pruning import recover_tokens
                                cond_recovered = recover_tokens(last_stage[:B], _topk_hw, pn, last_stage.device, per_frame=getattr(args, 'prune_per_frame', False))
                                uncond_recovered = recover_tokens(last_stage[B:], _topk_hw, pn, last_stage.device, per_frame=getattr(args, 'prune_per_frame', False))
                            else:
                                if getattr(args, 'prune_mode', 'prune') == 'merge':
                                    from infinity.models.st_token_pruning import unmerge_recover
                                    cond_recovered = unmerge_recover(last_stage[:B], _topk_hw, pn, pu_base=_pu_base[:B], beta=getattr(args, 'merge_beta', 0.3))
                                    uncond_recovered = unmerge_recover(last_stage[B:], _topk_hw, pn, pu_base=_pu_base[B:], beta=getattr(args, 'merge_beta', 0.3))
                                else:
                                    cond_recovered = partial_update_recover(last_stage[:B], _flat_idx, _pu_base[:B])
                                    uncond_recovered = partial_update_recover(last_stage[B:], _flat_idx, _pu_base[B:])
                        last_stage = torch.cat([cond_recovered, uncond_recovered], dim=0)
                        self._prune_pu_cache[('normal', si)] = last_stage.detach()
                    if getattr(args, 'prune_error_aware', False) and _do_prune:
                        _prev_counts = self._prune_count_cache.get(('normal', si), torch.zeros(1, pn[1]*pn[2], device=last_stage.device))
                        _pruned_mask = torch.ones(1, pn[1]*pn[2], device=last_stage.device)
                        _pruned_mask.scatter_(1, _topk_hw[:1], 0.0)
                        self._prune_count_cache[('normal', si)] = _prev_counts + _pruned_mask
                    # FastSTAR: cache features for next scale/clip scoring
                    if _scoring == 'faststar':
                        self._faststar_scale_cache[('normal', si)] = last_stage[:B].detach()
                        if last_repetition_step:
                            self._faststar_clip_cache[('normal', rel_si_in_one_clip)] = last_stage[:B].detach()
                    elif _scoring == 'fastvar':
                        self._fastvar_scale_cache[('normal', si)] = last_stage[:B].detach()
                        if last_stage.shape[0] > B:
                            self._fastvar_scale_cache[('normal_uncond', si)] = last_stage[B:].detach()
                    elif _scoring == 'v4':
                        self._faststar_scale_cache[('normal', si)] = last_stage[:B].detach()
                    elif _scoring == 'attn':
                        if last_repetition_step:
                            self._attn_clip_cache[('normal', rel_si_in_one_clip)] = last_stage[:B].detach()
                            if rel_si_in_one_clip in self._attn_imp_last:
                                self._attn_imp_clip_cache[rel_si_in_one_clip] = self._attn_imp_last[rel_si_in_one_clip]
                    logits_BlV = self.get_logits_during_infer(last_stage, is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(1/tau_list[si])
                    if cfg != 1:
                        if args.use_cfg:
                            logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                        elif args.use_apg:
                            pred_cond = logits_BlV[:B]
                            pred_uncond = logits_BlV[B:]
                            pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg, momentum_buffer=None, eta=0, norm_threshold=args.apg_norm_threshold)
                            logits_BlV = pred_guided
                    else:
                        logits_BlV = logits_BlV[:B]
                
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                probs_Bld = logits_BlV.softmax(dim=-1) # [B, thwd or thw4d, 2]
                _skip_code = getattr(args, 'prune_skip_code', 'zero')
                _p1_Bld = probs_Bld[..., 1].clone() if (_skip_code == 'expect' and self.num_of_label_value == 2) else None
                idx_Bld = torch.multinomial(probs_Bld.view(-1, self.num_of_label_value), num_samples=1, replacement=True, generator=rng).view(tmp_bs, -1) # [B, thwd or thw4d]
                probs_Bld = torch.gather(probs_Bld, dim=2, index=idx_Bld.unsqueeze(-1)).squeeze(-1)

                def Bld2Bthwd(item):
                    item = item.reshape(tmp_bs, tmp_seq_len, -1) # [B, thw, d or 4d]
                    item = item.reshape(B, pn[0], pn[1], pn[2], -1) # shape: [B, t, h, w, d] or [B, t, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        item = item.permute(0,1,4,2,3) # [B, t, 4d, h, w]
                        item = torch.nn.functional.pixel_shuffle(item, 2) # [B, t, d, 2h, 2w]
                        item = item.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
                    return item

                idx_Bld = Bld2Bthwd(idx_Bld)
                probs_Bld = Bld2Bthwd(probs_Bld)
                if _p1_Bld is not None:
                    _p1_Bld = Bld2Bthwd(_p1_Bld)
                # print(f'{si=} {repeat_idx=} idx_Bld.shape={idx_Bld.shape}')
                import os as _os
                if _os.environ.get('PRUNE_DIAG') and si >= 24 and repeat_idx == 0:
                    _ib = idx_Bld[0]
                    if _ib.shape[0] > 1:
                        _ta = (_ib[1:] == _ib[:-1]).float().mean().item()
                        _saw = (_ib[:,:,1:] == _ib[:,:,:-1]).float().mean().item()
                        _sah = (_ib[:,1:,:] == _ib[:,:-1,:]).float().mean().item()
                        _tf = (_ib[1:] == _ib[:-1]).all(dim=-1).float().mean().item()
                        _sf = (_ib[:,:,1:] == _ib[:,:,:-1]).all(dim=-1).float().mean().item()
                        print(f'[DIAG] si={si} pn={tuple(pn)} bit_agree T={_ta:.3f} Sw={_saw:.3f} Sh={_sah:.3f} | posmatch T={_tf:.3f} S={_sf:.3f}', flush=True)

                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[cum_scales+repeat_idx]
                # idx_Bld [B, t, h, w, d] or [B, t, 2h, 2w, d]
                if self.other_args.use_two_stage_lfq:
                    if pn[1] * pn[2] >= vae.quantizer.detail_scale_min_tokens:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    codes_exp = None
                    if _p1_Bld is not None:
                        # BSQ codes are +-1/sqrt(d) per element, so E[code] is the
                        # p-weighted interpolation of the all-0 and all-1 codes.
                        _c1 = lfq.indices_to_codes(torch.ones_like(idx_Bld), 'bit_label')
                        _c0 = lfq.indices_to_codes(torch.zeros_like(idx_Bld), 'bit_label')
                        codes_exp = _c0 + (_c1 - _c0) * _p1_Bld.permute(0, 4, 1, 2, 3).to(_c0.dtype)
                        codes_exp = interpolate(codes_exp, size=(self.vae_embed_dim, *target_pn), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                    codes = interpolate(codes, size=(self.vae_embed_dim, *target_pn), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes_exp = None
                    if _p1_Bld is not None:
                        _q = vae.quantizer.lfq_detail
                        _c1 = _q.indices_to_codes(torch.ones_like(idx_Bld), 'bit_label')
                        _c0 = _q.indices_to_codes(torch.zeros_like(idx_Bld), 'bit_label')
                        codes_exp = _c0 + (_c1 - _c0) * _p1_Bld.permute(0, 4, 1, 2, 3).to(_c0.dtype)
                        codes_exp = F.interpolate(codes_exp, size=target_pn, mode=vae.quantizer.z_interplote_up)
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] = F.interpolate(summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up)
                import os as _os_hf
                _hf = _os_hf.environ.get("HF_RECOVER")
                if _hf in ("nn", "blend") and _zf_topk_hw is not None:
                    _thh, _tww = target_pn[-2], target_pn[-1]
                    _phh, _pww = _zf_pn[1], _zf_pn[2]
                    _km = torch.zeros(_phh * _pww, device=codes.device)
                    _km[_zf_topk_hw[0]] = 1.0
                    _km = _km.view(1, 1, _phh, _pww)
                    if _phh != _thh or _pww != _tww:
                        _km = F.interpolate(_km, size=(_thh, _tww), mode="nearest")
                    _km = _km.reshape(_thh * _tww)
                    _keptT = (_km > 0.5).nonzero(as_tuple=True)[0]
                    _dropT = (_km <= 0.5).nonzero(as_tuple=True)[0]
                    if len(_keptT) > 0 and len(_dropT) > 0:
                        _kh = (_keptT // _tww).float(); _kw = (_keptT % _tww).float()
                        _dh = (_dropT // _tww).float(); _dw = (_dropT % _tww).float()
                        _dist = (_dh[:, None] - _kh[None]) ** 2 + (_dw[:, None] - _kw[None]) ** 2
                        _nnk = _keptT[_dist.argmin(dim=1)]
                        _src = torch.arange(_thh * _tww, device=codes.device)
                        _src[_dropT] = _nnk
                        _Bc, _Ec, _Tc = codes.shape[0], codes.shape[1], codes.shape[2]
                        _cf = codes.reshape(_Bc, _Ec, _Tc, _thh * _tww)
                        codes = _cf.gather(3, _src.view(1, 1, 1, -1).expand(_Bc, _Ec, _Tc, -1)).reshape(_Bc, _Ec, _Tc, _thh, _tww)
                        if _hf == "blend":
                            _a = float(_os_hf.environ.get("HF_ALPHA", "0.5"))
                            _bm = torch.ones(_thh * _tww, device=codes.device)
                            _bm[_dropT] = _a
                            codes = codes * _bm.view(1, 1, 1, _thh, _tww)
                elif _use_soft_zf and _zf_topk_hw is not None:
                    _szf_h, _szf_w = _zf_pn[1], _zf_pn[2]
                    _szf_hw = _szf_h * _szf_w
                    _szf_kept = _zf_topk_hw[0]
                    _szf_mask = torch.zeros(_szf_hw, device=codes.device)
                    _szf_mask[_szf_kept] = 1.0
                    _szf_non = (_szf_mask == 0).nonzero(as_tuple=True)[0]
                    if len(_szf_non) > 0:
                        _kh = (_szf_kept // _szf_w).float()
                        _kw = (_szf_kept % _szf_w).float()
                        _nh = (_szf_non // _szf_w).float()
                        _nw = (_szf_non % _szf_w).float()
                        _dd = ((_nh.unsqueeze(1) - _kh.unsqueeze(0))**2 + (_nw.unsqueeze(1) - _kw.unsqueeze(0))**2).sqrt()
                        _d_near = _dd.min(dim=1).values
                        _sigma = max(1.0, (_szf_hw / max(len(_szf_kept), 1)) ** 0.5 * 0.8)
                        _D_thresh = 1.5
                        _szf_mask[_szf_non] = (_d_near <= _D_thresh).float()
                    _szf_mask = _szf_mask.view(1, 1, _szf_h, _szf_w)
                    _th, _tw = target_pn[-2], target_pn[-1]
                    if _szf_h != _th or _szf_w != _tw:
                        _szf_mask = F.interpolate(_szf_mask, size=(_th, _tw), mode='nearest')
                    _szf_mask = _szf_mask.unsqueeze(2)
                    codes = codes * _szf_mask
                elif _faststar_zero_fill and _zf_topk_hw is not None:
                    from baselines.faststar import faststar_create_code_mask
                    _zf_mask = faststar_create_code_mask(_zf_topk_hw, _zf_pn, target_pn, codes.device)
                    if codes_exp is not None:
                        # skipped positions take beta * E[code] under the RC-recovered logits
                        _beta = float(getattr(args, 'prune_skip_code_beta', 1.0))
                        codes = codes * _zf_mask + _beta * codes_exp * (1 - _zf_mask)
                    else:
                        codes = codes * _zf_mask
                _oracle = getattr(args, 'prune_oracle', False)
                _oratios = getattr(args, 'oracle_ratios', None)
                import os as _os2
                _oscales_env = _os2.environ.get('ORACLE_SCALES', '')
                _oallowed = set(int(x) for x in _oscales_env.split(',') if x) if _oscales_env else None
                _oskip = bool(last_repetition_step and repeat_times > 1)
                if _oracle and _oratios and si in _oratios and (_oallowed is None or si in _oallowed) and not _oskip:
                    _ometric = _os2.environ.get('ORACLE_METRIC', 'conf')
                    _okr = _oratios[si]
                    _ph, _pw = pn[1], pn[2]
                    if _ometric == 'energy':
                        _ocodes_pn = lfq.indices_to_codes(idx_Bld, 'bit_label').float()
                        _osc = (_ocodes_pn ** 2).sum(dim=1).mean(dim=1)
                    elif _ometric == 'codehp':
                        _ocodes_pn = lfq.indices_to_codes(idx_Bld, 'bit_label').float()
                        _cmean = _ocodes_pn.mean(dim=2)
                        _clow = F.avg_pool2d(F.pad(_cmean, (1,1,1,1), mode='replicate'), 3, 1, 0)
                        _osc = ((_cmean - _clow) ** 2).sum(dim=1)
                    elif _ometric == 'random':
                        _ar = torch.arange(_ph*_pw, device=codes.device)
                        _osc = (((_ar*2654435761) % 100003).float()/100003.0).view(1,_ph,_pw)
                    else:
                        _osc = probs_Bld.float().mean(dim=(1, 4))
                    # _osc: [B, Hs, Ws] at idx_Bld resolution -> pool to (ph, pw)
                    _Bsz = _osc.shape[0]
                    _osc4 = _osc.unsqueeze(1)
                    if _osc.shape[1] != _ph or _osc.shape[2] != _pw:
                        _osc4 = F.adaptive_avg_pool2d(_osc4, (_ph, _pw))
                    _osc_flat = _osc4.reshape(_Bsz, _ph * _pw)
                    _okk = max(1, int(_ph * _pw * _okr))
                    _oflip = _os2.environ.get('ORACLE_FLIP', '0') == '1'
                    _, _otopk = _osc_flat.topk(_okk, dim=-1, largest=(not _oflip))
                    _otopk, _ = _otopk.sort(dim=-1)
                    from baselines.faststar import faststar_create_code_mask
                    _omask = faststar_create_code_mask(_otopk, pn, target_pn, codes.device)
                    if _os2.environ.get('ORACLE_DBG') == '1':
                        print(f'[ODBG] si={si} codes={tuple(codes.shape)} idxBld={tuple(idx_Bld.shape)} osc={tuple(_osc.shape)} pn=({_ph},{_pw}) mask={tuple(_omask.shape)} kk={_okk}', flush=True)
                    codes = codes * _omask
                summed_codes[-1] += codes
                if repeat_idx < repeat_times - 1:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down)
                    last_stage = self.embeds_codes2input(last_stage, bs//B)
            cum_scales += repeat_times
            if si < len(scale_schedule)-1:
                if scale_schedule[si][-2:] == scale_schedule[-1][-2:]:
                    if self.other_args.noise_input:
                        summed_codes.append(torch.randn((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    else:
                        summed_codes.append(torch.zeros((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    last_stage = summed_codes[-1]
                else:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down)
                last_stage = self.embeds_codes2input(last_stage, bs//B)
        summed_codes = torch.cat(summed_codes, dim=-3)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        if hasattr(self, '_stagevar_output_cache'):
            del self._stagevar_output_cache
        if hasattr(self, '_prune_pu_cache'):
            del self._prune_pu_cache
        if hasattr(self, '_faststar_scale_cache'):
            del self._faststar_scale_cache
        if hasattr(self, '_fastvar_scale_cache'):
            del self._fastvar_scale_cache
        if hasattr(self, '_sparsevar_mse_cache'):
            del self._sparsevar_mse_cache
        if hasattr(self, '_sparsevar_pn_cache'):
            del self._sparsevar_pn_cache
        if hasattr(self, '_faststar_clip_cache'):
            del self._faststar_clip_cache
        if return_summed_code_only:
            return summed_codes
        else:
            if low_vram_mode: vae.to('cuda')
            img = self.summed_codes2images(vae, summed_codes)
            return idx_Bl_list, img


    @torch.no_grad()
    def ar_infer_infinity_star_interact(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        context_info=None,
        return_summed_code_only=False,
        mode='',
        former_clip_features=None,
        first_frame_features=None,
        semantic_scale_ind = 7,
        detail_frame_inds = [18,19],
        **kwargs,
    ):   # returns List[idx_Bl]
        from infinity.schedules.infinity_star_interact import interpolate
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_apg + args.use_cfg == 1
        device = label_B_or_BLT[0].device
        if g_seed is None: 
            rng = None
        else:
            self.rng = torch.Generator(device=device)
            self.rng.manual_seed(g_seed)
            rng = self.rng
        
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = label_B_or_BLT[-1] # self.text_maxlen # kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        prefix_tokens, _ = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT, vae_scale_schedule, text_token_only=False, text_maxlen_this_iter=text_maxlen_this_iter)
        bs = prefix_tokens.shape[0]

        ca_kv, cond_BD_or_gss, attn_mask = None, None, None
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))
        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
        assert len(image_scale_repetition) == scales_in_one_clip, f'{len(image_scale_repetition)} != {scales_in_one_clip}'
        total_steps = image_scale_repetition.sum() + video_scale_repetition.sum() * (len(scale_schedule)//len(video_scale_repetition)-1) + 1 # +1 is prefix text token forward step
        if mode == 'second_v_clip':
            total_steps += 2
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks

        noise_shape = vae_scale_schedule[0]
        if self.other_args.noise_input:
            noise = torch.randn((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)
        else:
            noise = torch.zeros((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype, device=prefix_tokens.device)
        
        summed_codes = [noise[0:1]]
        sos_token = self.embeds_codes2input(noise, bs//1)
        # text tokens forward
        rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter]
        last_stage = prefix_tokens
        for block_idx, b in enumerate(block_chunks):
            last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=f't0', context_info=context_info, last_repetition_step=True)
        pbar.update(1)

        ref_text_scale_inds = ['t0']

        # visual condition forward
        if mode == 'second_v_clip':
            assert former_clip_features.shape[-3] == 21
            former_clip_features = former_clip_features[:,:,1:]
            last_stage = F.interpolate(former_clip_features, size=(20, *vae_scale_schedule[semantic_scale_ind][-2:]), mode=vae.quantizer.z_interplote_down)
            rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule[-1], last_stage.shape[-3:], list(range(1, 21)), 800, device)
            last_stage = self.embeds_codes2input(last_stage, bs//B)
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=f'semantic_condition', context_info=context_info, last_repetition_step=True)
            pbar.update(1)

            last_stage = torch.cat([first_frame_features, former_clip_features[:,:,detail_frame_inds]], dim=2)
            rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule[-1], last_stage.shape[-3:], [0]+[item+1 for item in detail_frame_inds], 801, device)
            last_stage = self.embeds_codes2input(last_stage, bs//B)
            # anchors for IDS: the conditioning frames this clip continues from
            self._ip_cond_hidden = last_stage[:B].detach()
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=f'detail_condition', context_info=context_info, last_repetition_step=True)
            pbar.update(1)

            ref_text_scale_inds.extend(['semantic_condition', 'detail_condition'])

        # visual tokens forward
        last_stage = sos_token
        cum_scales = 0
        # IDS/RC pruning for interactive generation (off unless args.prune_scoring == 'attn')
        _ip_on = getattr(args, 'prune_scoring', None) == 'attn'
        _ip_ratios = getattr(args, 'prune_ratios_per_scale', None) or {}
        _ip_clip_h, _ip_clip_imp, _ip_imp_last, _ip_pu = {}, {}, {}, {}
        _ip_att = None
        _ip_cond = getattr(self, '_ip_cond_hidden', None) if mode == 'second_v_clip' else None
        self._ip_cond_hidden = None
        if _ip_on:
            for _bm in self.modules():
                if _bm.__class__.__name__ == 'SelfAttention':
                    _ip_att = _bm; break
        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            rel_si_in_one_clip = si % scales_in_one_clip
            if si < scales_in_one_clip: # image
                repeat_times = image_scale_repetition[rel_si_in_one_clip]
                target_pn = vae_scale_schedule[first_full_spatial_size_scale_index]
            else:
                repeat_times = video_scale_repetition[rel_si_in_one_clip]
                target_pn = vae_scale_schedule[-1]
            cfg = cfg_list[si]
            infer_repeat_times = min(repeat_times, args.max_repeat_times)
            for repeat_idx in range(infer_repeat_times):
                frame_ss, frame_ee = context_info[si]['frame_ss'], context_info[si]['frame_ee']
                rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule[-1], scale_schedule[si], list(range(frame_ss, frame_ee)), cum_scales+repeat_idx, device)
                last_repetition_step = (repeat_idx == (infer_repeat_times-1))
                _ip_keep = _ip_ratios.get(si, 1.0)
                _ip_do = _ip_on and _ip_keep < 1.0 and pn[1] * pn[2] >= getattr(args, 'prune_token_threshold', 10**9)
                _ip_zf = None
                rope_fwd = rope_cache
                if _ip_do:
                    from infinity.models.st_token_pruning import compute_attention_importance, select_attention_tokens
                    _ck = _ip_att.cached_k
                    _text_k = _ck[0] if (isinstance(_ck, list) and len(_ck) > 0) else _ck.get('t0', _ck.get(0, None))
                    _full_stage = last_stage
                    _imp = compute_attention_importance(
                        last_stage[:B], pn, _ip_att, text_k=_text_k, text_lens=None,
                        prev_clip_hidden=_ip_cond if getattr(args, 'prune_attn_temporal_mode', 'column') == 'anchor' else None,
                        w_spatial=getattr(args, 'prune_w_spatial', 1.0),
                        w_temporal=getattr(args, 'prune_w_temporal', 1.0),
                        w_text=getattr(args, 'prune_w_text', 1.0),
                        fusion_p=getattr(args, 'prune_fusion_p', 2.0),
                        max_query_tokens=getattr(args, 'prune_attn_max_queries', None),
                        rope_cache=rope_cache,
                        per_frame=getattr(args, 'prune_per_frame', False),
                        text_query_gate=getattr(args, 'prune_attn_text_query_gate', 0.0),
                        temporal_mode=getattr(args, 'prune_attn_temporal_mode', 'column'),
                        prev_importance=None,
                        temporal_anchors=getattr(args, 'prune_attn_temporal_anchors', 1024),
                        random_select=getattr(args, 'prune_attn_random_select', False),
                    )
                    _ip_imp_last[rel_si_in_one_clip] = _imp.detach()
                    _cond, rope_fwd, _flat_idx, _topk_hw = select_attention_tokens(last_stage[:B], rope_cache, pn, _imp, _ip_keep)
                    _unc = last_stage[B:].gather(1, _flat_idx.unsqueeze(-1).expand(-1, -1, last_stage.shape[-1]))
                    last_stage = torch.cat([_cond, _unc], dim=0)
                for block_idx, b in enumerate(block_chunks):
                    last_stage = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_fwd, scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step, ref_text_scale_inds=ref_text_scale_inds)
                if _ip_do:
                    from infinity.models.st_token_pruning import recover_attention_tokens
                    _base = _ip_pu.get(si, _full_stage)
                    _kw = dict(per_frame=getattr(args, 'prune_per_frame', False), mode=getattr(args, 'prune_recover_mode', 'pu'),
                               n_neighbors=getattr(args, 'prune_recover_neighbors', 4), residual_scale=getattr(args, 'prune_recover_alpha', 1.0))
                    _rc = recover_attention_tokens(last_stage[:B], _flat_idx, _base[:B], _full_stage[:B], pn, _topk_hw, **_kw)
                    _ru = recover_attention_tokens(last_stage[B:], _flat_idx, _base[B:], _full_stage[B:], pn, _topk_hw, **_kw)
                    last_stage = torch.cat([_rc, _ru], dim=0)
                    _ip_pu[si] = last_stage.detach()
                    if getattr(args, 'prune_recover_code_mask', True):
                        _ip_zf = (_topk_hw, pn)
                if _ip_on and last_repetition_step:
                    _ip_clip_h[rel_si_in_one_clip] = last_stage[:B].detach()
                    if rel_si_in_one_clip in _ip_imp_last:
                        _ip_clip_imp[rel_si_in_one_clip] = _ip_imp_last[rel_si_in_one_clip]
                logits_BlV = self.get_logits_during_infer(last_stage, is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(1/tau_list[si])
                if cfg != 1:
                    # print(f'add cfg on add_cfg_on_logits')
                    if args.use_cfg:
                        logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                    elif args.use_apg:
                        pred_cond = logits_BlV[:B]
                        pred_uncond = logits_BlV[B:]
                        pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg, momentum_buffer=None, eta=0, norm_threshold=args.apg_norm_threshold)
                        # pred_guided = cfg * pred_cond + (1-cfg) * pred_uncond
                        logits_BlV = pred_guided
                else:
                    logits_BlV = logits_BlV[:B]
                
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                probs_Bld = logits_BlV.softmax(dim=-1) # [B, thwd or thw4d, 2]
                _skip_code = getattr(args, 'prune_skip_code', 'zero')
                _p1_Bld = probs_Bld[..., 1].clone() if (_skip_code == 'expect' and self.num_of_label_value == 2) else None
                idx_Bld = torch.multinomial(probs_Bld.view(-1, self.num_of_label_value), num_samples=1, replacement=True, generator=rng).view(tmp_bs, -1) # [B, thwd or thw4d]
                probs_Bld = torch.gather(probs_Bld, dim=2, index=idx_Bld.unsqueeze(-1)).squeeze(-1)

                def Bld2Bthwd(item):
                    item = item.reshape(tmp_bs, tmp_seq_len, -1) # [B, thw, d or 4d]
                    item = item.reshape(B, pn[0], pn[1], pn[2], -1) # shape: [B, t, h, w, d] or [B, t, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        item = item.permute(0,1,4,2,3) # [B, t, 4d, h, w]
                        item = torch.nn.functional.pixel_shuffle(item, 2) # [B, t, d, 2h, 2w]
                        item = item.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
                    return item

                idx_Bld = Bld2Bthwd(idx_Bld)
                probs_Bld = Bld2Bthwd(probs_Bld)
                if _p1_Bld is not None:
                    _p1_Bld = Bld2Bthwd(_p1_Bld)

                if si < gt_leak:
                    acc = (idx_Bld==gt_ls_Bl[cum_scales+repeat_idx]).float().mean() * 100.
                    idx_Bld = gt_ls_Bl[cum_scales+repeat_idx]
                    print(f'{si=} {repeat_idx=} idx_Bld.shape={idx_Bld.shape} {acc=}%')

                # idx_Bld [B, t, h, w, d] or [B, t, 2h, 2w, d]
                if self.other_args.use_two_stage_lfq:
                    if si >= args.semantic_scales:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    codes_exp = None
                    if _p1_Bld is not None:
                        # BSQ codes are +-1/sqrt(d) per element, so E[code] is the
                        # p-weighted interpolation of the all-0 and all-1 codes.
                        _c1 = lfq.indices_to_codes(torch.ones_like(idx_Bld), 'bit_label')
                        _c0 = lfq.indices_to_codes(torch.zeros_like(idx_Bld), 'bit_label')
                        codes_exp = _c0 + (_c1 - _c0) * _p1_Bld.permute(0, 4, 1, 2, 3).to(_c0.dtype)
                        codes_exp = interpolate(codes_exp, size=(self.vae_embed_dim, *target_pn), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                    codes = interpolate(codes, size=(self.vae_embed_dim, *target_pn), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes_exp = None
                    if _p1_Bld is not None:
                        _q = vae.quantizer.lfq_detail
                        _c1 = _q.indices_to_codes(torch.ones_like(idx_Bld), 'bit_label')
                        _c0 = _q.indices_to_codes(torch.zeros_like(idx_Bld), 'bit_label')
                        codes_exp = _c0 + (_c1 - _c0) * _p1_Bld.permute(0, 4, 1, 2, 3).to(_c0.dtype)
                        codes_exp = F.interpolate(codes_exp, size=target_pn, mode=vae.quantizer.z_interplote_up)
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] = F.interpolate(summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up)
                if _ip_zf is not None:
                    from baselines.faststar import faststar_create_code_mask
                    codes = codes * faststar_create_code_mask(_ip_zf[0], _ip_zf[1], target_pn, codes.device)
                summed_codes[-1] += codes
                if repeat_idx < repeat_times - 1:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down)
                    last_stage = self.embeds_codes2input(last_stage, bs//B)
                pbar.update(1)
            cum_scales += repeat_times
            if si < len(scale_schedule)-1:
                if scale_schedule[si][-2:] == scale_schedule[-1][-2:]:
                    if self.other_args.noise_input:
                        summed_codes.append(torch.randn((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    else:
                        summed_codes.append(torch.zeros((B, summed_codes[-1].shape[1], *vae_scale_schedule[si+1]), device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    last_stage = summed_codes[-1]
                else:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down)
                last_stage = self.embeds_codes2input(last_stage, bs//B)
        summed_codes = torch.cat(summed_codes, dim=-3)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        if mode == 'second_v_clip':
            this_clip_frames = summed_codes.shape[2] * 4
            summed_codes = torch.cat([former_clip_features, summed_codes], dim=-3)
            img = self.summed_codes2images(vae, summed_codes) # [bs, t, h, w, 3]
            img = img[:,-this_clip_frames:]
            summed_codes = summed_codes[:,:,-21:]
            assert summed_codes.shape[2] == 21, f'wrong shape: {summed_codes.shape=}'
        else:
            img = self.summed_codes2images(vae, summed_codes)
        
        if low_vram_mode: vae.to('cuda')
        return summed_codes, img 
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        returns_vemb=0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        **kwargs,
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_cfg + args.use_apg == 1
        device = label_B_or_BLT[0].device
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = self.text_maxlen
        last_stage, lens, _ = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT, args.input_noise, vae_scale_schedule)
        bs = last_stage.shape[0]
        ca_kv, cond_BD_or_gss = None, None
        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            visual_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule, si, device, args)
            if si == 0:
                rope_cache = torch.cat([self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:text_maxlen_this_iter], visual_rope_cache], dim=4)
            else:
                rope_cache = visual_rope_cache
            attn_mask = torch.ones((last_stage.shape[0], 1, last_stage.shape[1], text_maxlen_this_iter+np.array(pn).prod()), device=last_stage.device).bool() # [bs, q_heads, q_len, all_k_len], here set q_heads=1 for broadcasting
            assert len(attn_mask) == len(lens)              
            for tmp_i, le in enumerate(lens):
                attn_mask[tmp_i, :, :, le:text_maxlen_this_iter] = False
                if si == 0:
                    attn_mask[tmp_i, :, :text_maxlen_this_iter, text_maxlen_this_iter:] = False
            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            for block_idx, b in enumerate(self.block_chunks):
                for m in b.module:
                    last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, scale_ind=si)
            if si == 0:
                last_stage = last_stage[:, text_maxlen_this_iter:]
            # import pdb; pdb.set_trace()
            if cfg != 1:
                # print(f'add cfg on add_cfg_on_logits')
                logits_BlV = self.get_logits(last_stage).mul(1/tau_list[si])
                if args.use_cfg:
                    logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
                elif args.use_apg:
                    pred_cond = logits_BlV[:B]
                    pred_uncond = logits_BlV[B:]
                    pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg, momentum_buffer=None, eta=0, norm_threshold=10)
                    # pred_guided = cfg * pred_cond + (1-cfg) * pred_uncond
                    logits_BlV = pred_guided
            else:
                logits_BlV = self.get_logits(last_stage[:B]).mul(1/tau_list[si])
            if self.num_of_label_value == 1:
                idx_Bld = logits_BlV
            elif self.num_of_label_value > 1:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            elif self.num_of_label_value == 0:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            assert returns_vemb
            if si < gt_leak:
                idx_Bld = gt_ls_Bl[si]
            else:
                idx_Bld = idx_Bld.reshape(B, pn[0], pn[1], pn[2], -1) # shape: [B, t, h, w, d] or [B, t, h, w, 4d]
                if self.apply_spatial_patchify: # unpatchify operation
                    idx_Bld = idx_Bld.permute(0,1,4,2,3) # [B, t, 4d, h, w]
                    idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, t, d, 2h, 2w]
                    idx_Bld = idx_Bld.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
                # idx_Bld [B, t, h, w, d] or [B, t, 2h, 2w, d]

            # idx_Bld_list.append(idx_Bld)
            if self.num_of_label_value == 1:
                if si < gt_leak:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, t, h, w] or [B, d, t, 2h, 2w]
                else:
                    codes = idx_Bld.permute(0,4,1,2,3)
            else:
                codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, t, h, w] or [B, d, t, 2h, 2w]
            if vae_scale_schedule[si] != vae_scale_schedule[-1]:
                codes = F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            summed_codes += codes
            if si < len(scale_schedule)-1:
                last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_down) # [B, d, t, h, w] or [B, d, t, 2h, 2w]
                if self.apply_spatial_patchify: # patchify operation
                    last_stage = last_stage.permute(0,2,1,3,4) # [B, t, d, 2h, 2w]
                    last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, t, 4d, h, w]
                    last_stage = last_stage.permute(0,2,1,3,4) # [B, 4d, t, h, w]
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, t*h*w] or [B, 4d, t*h*w]
                last_stage = torch.permute(last_stage, [0,2,1]) # [B, t*h*w, d] or [B, t*h*w, 4d]
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        if low_vram_mode: vae.to('cuda')
        img = self.summed_codes2images(vae, summed_codes)
        return ret, idx_Bl_list, img
    
    def summed_codes2images(self, vae, summed_codes):
        t1 = time.time()

        img = vae.decode(summed_codes, slice=True)
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        img = img.permute(0,2,3,4,1) # [bs, 3, t, h, w] -> [bs, t, h, w, 3]
        img = img.mul_(255).to(torch.uint8).flip(dims=(4,))

        # Replace the VAE warm-up frame only when a later video frame exists.
        if img.shape[1] > 1:
            img[:, 0:1, :, :, :] = img[:, 1:2, :, :, :]
            
        print(f'Decode takes {time.time()-t1:.1f}s')
        return img

    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
        
        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)
        
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(self):
        if self.arch == 'qwen':
            std = 0.02
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
        else:
            raise ValueError(f'Unknown arch {self.arch}')
    
    def extra_repr(self):
        return f''
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # sa+ca, mlp
    s += w**2 * 6       # saln
    s += 4096 * w       # pred
    s += 32 * w         # we
    
    Ct5 = 4096
    s += Ct5*w * 4      # T5 attn pool
    s += Ct5*w + w*w    # T5 mlp
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def infinity_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa2b(depth=28, block_chunks=7, embed_dim=2560, num_heads=2560//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa8b(depth=42, block_chunks=7, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa14b(depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, drop_path_rate=0.1, mlp_ratio=3.4, **kwargs): 
    return Infinity(
        depth=depth, 
        block_chunks=block_chunks, 
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        mlp_ratio=mlp_ratio, 
        drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )
    # (depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, num_key_value_heads=5120//128//4, drop_path_rate=0, **kwargs)

@register_model
def infinity_sa12b(depth=60, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa16b(depth=42, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_v2b(depth=32, embed_dim=2016, num_heads=2016//126, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_8b(depth=40, block_chunks=1, embed_dim=3584, num_heads=3584//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_qwen7b(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128//4, mlp_ratio=12288/4096, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        depth=depth, 
        block_chunks=block_chunks,
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        num_key_value_heads=num_key_value_heads, 
        mlp_ratio=mlp_ratio, 
        drop_path_rate=drop_path_rate, 
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen8b(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128//4, mlp_ratio=4, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen_wide14b(depth=36, block_chunks=6, embed_dim=5632, num_heads=5632//128, num_key_value_heads=5632//128//4, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen13bMHA(depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, num_key_value_heads=5120//128, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=True,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen2_2b(depth=28, block_chunks=7, embed_dim=2304, num_heads=2304//128, num_key_value_heads=2304//128, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen0b(depth=4, block_chunks=2, embed_dim=512, num_heads=512//128, num_key_value_heads=512//128, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen2_30b(depth=54, block_chunks=27, embed_dim=6144, num_heads=6144//128, num_key_value_heads=6144//128//4, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4, #mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen14b(depth=48, block_chunks=24, embed_dim=4608, num_heads=4608//128, num_key_value_heads=4608//128//4, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_20b(depth=58, embed_dim=4608, num_heads=4608//128, drop_path_rate=0.25, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

# model configuration for scaling Infinity transformer
@register_model
def infinity_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer16(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer24(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer32(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer40(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer48(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
