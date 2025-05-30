from typing import Dict, Optional, Tuple,List, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.init import normal_
import numpy as np

from specdetr.specdetrpositionalencoding import SinePositionalEncoding
from specdetr.specdetrencoder import SpecDetrTransformerEncoder
from specdetr.specdetr_atten import MultiScaleDeformableAttention_1 as MultiScaleDeformableAttention
from specdetr.specdetrbackbone import No_backbone_ST




class SpecDetrWrapper(nn.Module): #Bruke BaseModule eller nn.Module??
    def __init__(self, backbone, positional_encoding, encoder, num_feature_levels): #GJØR DENNE RIKTIG!!
        super().__init__()
        self.encoder = encoder
        self.encoder_layers_num = encoder["num_layers"]
        self.positional_encoding = positional_encoding
        self.backbone = backbone
        self.num_feature_levels = num_feature_levels
        self.with_neck = False
        self.save_id = 0
        self._init_layers() #Dette skjedde originalt i DeformableDetr eller DetectionTransformer
        self.init_weights() #Bør vi gjøre dette her? Tja why not


    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head.""" #SIKE VI INITIALISERER BACKBONE ELLER?
        self.backbone = No_backbone_ST(**self.backbone)  
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        if self.encoder_layers_num > 0:
            self.encoder = SpecDetrTransformerEncoder(**self.encoder)
        self.embed_dims = self.encoder.embed_dims
        
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        
    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""

        if hasattr(self.backbone, 'init_weights'):
            print("BACKBONE INIT WEIGHTS!")
            self.backbone.init_weights()

        if self.encoder_layers_num>0:
            for p in self.encoder.parameters(): ##FUNKER DETTE? GJORDE litt modifikasjoner
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        normal_(self.level_embed)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs) ##MÅ initialisere backbone noe sted. I __init__? Eller __init_layers?
        if self.with_neck: ##Skal jeg bare initialize with_neck til False by default i __init__?
            x = self.neck(x)
        self.save_id += 1

        return x

    def forward(self, 
            batch_inputs: Tensor) -> Tensor: ##OG returnerer den riktig type nå?
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        encoder_inputs_dict = self.pre_transformer(img_feats) ##Istede for å ta inn batch_data_samples, skal vi bare bruke self.batch_input_shape inni pre_transforer-method?
        encoder_outputs = self.forward_encoder(**encoder_inputs_dict) ## MÅ skjønne encoder outputs dict!! Tror jeg bare outputter memory fra forward_encoder, som er queriesa. Så må jeg average her
        
        results = encoder_outputs.mean(dim=1)  

        return results

    

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor]) -> Dict:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = mlvl_feats[0].size(0)
        batch_input_shape = mlvl_feats[0].shape[2:] #Siden backbone IKKE endrer på størrelsen, OG alle input images i en batch er lik str fra start, så setter vi bare batch_input_shape til dette. M
        #MEEEN, BØR VI EGT BRUKE BILDESTØRRELSEN FØR DATAAUGMENTATION? NÆ TROKKE DET
    
        #assert self.batch_input_shape is not None #Venta, dette er avhengig av om batchen er local eller global!! Kan vel ikke initialisere student slik! Kan gjøre det i forward-pass inni multicropwrapper??
        #img_shape_list = [sample.img_shape for sample in batch_data_samples] #Her må jeg hente img_shape for alle i batchen, uten å bruke batch_data_samples. Skal jeg sende en liste inn i wrapperen?? OG -> hvordan blir dette mtp DINOAugmentation? Kan jeg bruke mlvl_feats bare??
        img_shape_list = [mlvl_feats[0].shape[2:] for _ in range(batch_size)] 
        input_img_h, input_img_w = batch_input_shape
        masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            mask = mask.flatten(1)
            spatial_shape = (h, w)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        mask_flatten = torch.cat(mask_flatten, 1)

        spatial_shapes = torch.as_tensor(  # (num_level, 2)
            spatial_shapes,
            dtype=torch.long,
            device=feat_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(  # (bs, num_level, 2)
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict
    

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Tensor:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        if self.encoder_layers_num > 0:
            memory = self.encoder( #memory har shape (bs, num_queries, dim). HVA ER NUM_QUERIES? SAMME SOM num_feat_points!!! JA
                query=feat,
                query_pos=feat_pos,
                key_padding_mask=feat_mask,  # for self_attn
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios)
        return memory
    

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Get the valid radios of feature map in a level.

        .. code:: text

                    |---> valid_W <---|
                 ---+-----------------+-----+---
                  A |                 |     | A
                  | |                 |     | |
                  | |                 |     | |
            valid_H |                 |     | |
                  | |                 |     | H
                  | |                 |     | |
                  V |                 |     | |
                 ---+-----------------+     | |
                    |                       | V
                    +-----------------------+---
                    |---------> W <---------|

          The valid_ratios are defined as:
                r_h = valid_H / H,  r_w = valid_W / W
          They are the factors to re-normalize the relative coordinates of the
          image to the relative coordinates of the current level feature map.

        Args:
            mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

        Returns:
            Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio



