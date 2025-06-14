# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional, no_type_check
import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, constant_init, xavier_init
from mmengine.registry import MODELS
from mmengine.utils import deprecated_api_warning
from torch.autograd.function import Function, once_differentiable
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from mmcv.utils import ext_loader


ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class MultiScaleDeformableAttnFunction(Function):

    @staticmethod
    def forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        """GPU/MLU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (torch.Tensor): The step used in image to column.

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step

        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of sampling_locations, attention_weights
        # (float32), but "value" is cast to float16, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "value",
        # we cast sampling_locations and attention_weights to
        # temporarily support fp16 and amp whatever the
        # pytorch version is.
        sampling_locations = sampling_locations.type_as(value)
        attention_weights = attention_weights.type_as(value)

        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """GPU/MLU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


def multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention_1(nn.Module):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Default: 1.0.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 local_attn_type: str = 'initial_version',  # 'fix_same_orientation'  'fix_same_distance' 'initial_version' bbox_fine'
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[mmengine.ConfigDict] = None,
                 value_proj_ratio: float = 1.0):
        #super().__init__(init_cfg)
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        print("EMBEDDED DIMENSIONS: ", embed_dims)
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.local_attn_type = local_attn_type
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        if local_attn_type == 'initial_version' or self.local_attn_type == 'bbox_fine':
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
        else:
        #修改
            self.sampling_offsets = torch.zeros(num_heads * num_levels * num_points * 2)

        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()


    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        if self.local_attn_type == 'initial_version' or self.local_attn_type == 'bbox_fine':
            constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        if self.local_attn_type == 'initial_version' or self.local_attn_type == 'bbox_fine':
            thetas = torch.arange(
                self.num_heads, dtype=torch.float32,
                device=device) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                         grid_init.abs().max(-1, keepdim=True)[0]).view(
                             self.num_heads, 1, 1,
                             2).repeat(1, self.num_levels, self.num_points, 1)
            for i in range(self.num_points):
                grid_init[:, :, i, :] *= i + 1
            self.sampling_offsets.bias.data = grid_init.view(-1)
            # self.sampling_offsets.bias.data = grid_init.view(-1)
            # constant_init(self.attention_weights, val=0., bias=0.)
            # xavier_init(self.value_proj, distribution='uniform', bias=0.)
            # xavier_init(self.output_proj, distribution='uniform', bias=0.)
            # self._is_init = True

        elif self.local_attn_type == 'fix_same_orientation':
            # 每个head四个点在同一方向上
            thetas = torch.arange(
                self.num_heads, dtype=torch.float32,
                device=device) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                         grid_init.abs().max(-1, keepdim=True)[0]).view(
                             self.num_heads, 1, 1,
                             2).repeat(1, self.num_levels, self.num_points, 1)
            for i in range(self.num_points):
                grid_init[:, :, i, :] *= i + 1
            self.sampling_offsets1 = grid_init.view(-1)
        elif self.local_attn_type == 'fix_same_distance':
            # changed by lzx
            # 每个head四个点与中心点距离相同
            assert  self.num_points == 4
            grid_init = torch.zeros(self.num_heads, 1, self.num_points, 2)
            grid_init[0, :, :, :] = torch.Tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])
            grid_init[1, :, :, :] = torch.Tensor([[1,1],[-1,1],[-1,-1],[1,-1]])
            for i in range(2, self.num_heads):
                if i % 2 == 0:
                    grid_init[i, :, :, :] = grid_init[0, :, :, :] * (i//2+1)
                else:
                    grid_init[i, :, :, :] = grid_init[1, :, :, :] * (i//2+1)
            grid_init = grid_init.repeat(1, self.num_levels, 1, 1)
        self.sampling_offsets1 = grid_init.view(-1)
        # 每个head四个点与中心点螺旋延伸
        # assert  self.num_points == 4
        # grid_init = torch.zeros(self.num_heads, 1, self.num_points, 2)
        # grid_init_old = torch.Tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])
        # grid_init_even = torch.Tensor([[1,1],[-1,1],[-1,-1],[1,-1]])
        # for i in range(self.num_heads):
        #     if i % 2 == 0:
        #         for p_i in range(4):
        #             grid_init[i, :, p_i, :] = (grid_init_old[p_i, :]) * ((i//2+p_i)%4+1)
        #     else:
        #         for p_i in range(4):
        #             grid_init[i, :, p_i, :] = (grid_init_even[p_i, :]) * ((i//2+p_i)%4+1)
        #         grid_init = grid_init.repeat(1, self.num_levels, 1, 1)


        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        if self.local_attn_type == 'initial_version'or self.local_attn_type == 'bbox_fine':
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            # sampling_offsets = self.sampling_offsets(torch.ones_like(query[0,0,:])).view(self.num_heads, self.num_levels, self.num_points, 2).repeat( bs, num_query, 1,1,1,1)  
        else:
        #     # 修改部分：将原先的 sampling_offsets 修改为 sampling_offsets_weight
            sampling_offsets = self.sampling_offsets1.view(
                1, 1, self.num_heads, self.num_levels, self.num_points, 2)
            sampling_offsets =sampling_offsets.repeat(
                bs, num_query, 1, 1, 1, 1)
            sampling_offsets = sampling_offsets.to(query.device)
       
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            if self.local_attn_type == 'initial_version':
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :] \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
            else:     
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :] \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            # ori
            # sampling_locations = reference_points[:, :, None, :, None, :2] \
            #     + sampling_offsets / self.num_points \
            #     * reference_points[:, :, None, :, None, 2:] \
            #     * 0.5
            if self.local_attn_type == 'initial_version':
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                    + sampling_offsets / self.num_points \
                    * reference_points[:, :, None, :, None, 2:] \
                    * 0.5
            elif self.local_attn_type == 'bbox_fine':
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                    + reference_points[:, :, None, :, None, 2:] *0.5*0.6+ sampling_offsets / self.num_points \
                    * reference_points[:, :, None, :, None, 2:] *0.5*0.8
            else:    
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                    + reference_points[:, :, None, :, None, 2:] * 0.25+ sampling_offsets / self.num_points \
                    * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
            # output = multi_scale_deformable_attn_pytorch(
            #     value, spatial_shapes, sampling_locations, attention_weights)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity