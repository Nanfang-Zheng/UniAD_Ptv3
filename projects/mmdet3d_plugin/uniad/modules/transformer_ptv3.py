import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.transformer import DetrTransformerDecoderLayer, TransformerLayerSequence
from projects.mmdet3d_plugin.uniad.modules.transformer import PerceptionTransformer
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class UniADPTv3Decoder(TransformerLayerSequence):
    """
    Decoder for PTv3-based UniAD.
    Mimics DetectionTransformerDecoder but supports 3D reference points passing.
    """
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(UniADPTv3Decoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                # Extra args
                indexer=None,
                **kwargs):
        
        output = query
        intermediate = []
        intermediate_reference_points = []
        
        for lid, layer in enumerate(self.layers):
            # Pass BOTH 2D and 3D reference points
            # 1. Prepare 2D for Visual (CustomMSDeformableAttention)
            # reference_points: (BS, N, 3) -> (BS, N, 1, 2)
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            
            # 2. Pass to Layer
            # We pass `reference_points` as `reference_points_3d` or similar via kwargs
            # Or we rely on the layer signature to accept `reference_points` as the 2D one 
            # and we pass 3D as a separate arg.
            # UniADPTv3DecoderLayer expects `reference_points` to be passed to `self.attentions[1]`.
            # `CustomMSDeformableAttention` expects `reference_points` to be (BS, N, L, 2).
            # So `reference_points` arg MUST be the 2D one.
            
            # We pass the original 3D points as `reference_points_3d`.
            
            output = layer(
                output,
                *args,
                reference_points=reference_points_input, # For Visual (Standard)
                reference_points_3d=reference_points,    # For LiDAR (New)
                key_padding_mask=key_padding_mask,
                indexer=indexer,
                **kwargs)
            
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 3
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        else:
            return output, reference_points

@TRANSFORMER_LAYER.register_module()
class UniADPTv3DecoderLayer(DetrTransformerDecoderLayer):
    def __init__(self, 
                 lidar_cross_attn_cfg=None,
                 fusion_cfg=None,
                 embed_dims=256,
                 **kwargs):
        super(UniADPTv3DecoderLayer, self).__init__(**kwargs)
        
        # LiDAR Cross Attention
        if lidar_cross_attn_cfg is None:
            lidar_cross_attn_cfg = dict(
                embed_dim=embed_dims,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        else:
             if 'embed_dims' in lidar_cross_attn_cfg:
                 lidar_cross_attn_cfg['embed_dim'] = lidar_cross_attn_cfg.pop('embed_dims')
             if 'embed_dim' not in lidar_cross_attn_cfg:
                 lidar_cross_attn_cfg['embed_dim'] = embed_dims

        self.lidar_cross_attn = nn.MultiheadAttention(**lidar_cross_attn_cfg)
        self.lidar_norm = nn.LayerNorm(lidar_cross_attn_cfg['embed_dim'])
        self.lidar_dropout = nn.Dropout(lidar_cross_attn_cfg.get('dropout', 0.1))
        
        # Fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(),
            nn.Linear(embed_dims // 2, 2), # alpha_cam, alpha_lidar
            nn.Softmax(dim=-1)
        )

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                # Arguments
                indexer=None,
                reference_points=None, # This is the 2D input from Decoder (BS, N, 1, 2)
                reference_points_3d=None, # This is the 3D input from Decoder (BS, N, 3)
                **kwargs):
        
        # 1. Self Attention (Standard)
        query = self.attentions[0](
            query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            **kwargs)
        
        query = self.norms[0](query)
        
        # 2. Camera Cross Attention (Standard)
        # We pass reference_points (2D) here
        query_cam = self.attentions[1](
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            reference_points=reference_points,
            **kwargs)
        
        # 3. LiDAR Cross Attention
        query_lidar = query 
        
        if indexer is not None and reference_points_3d is not None:
            # Use reference_points_3d (BS, N, 3)
            bs, num_query, _ = reference_points_3d.shape
            
            ref_pts_flat = reference_points_3d.reshape(-1, 3)
            batch_idx = torch.arange(bs, device=query.device).unsqueeze(1).repeat(1, num_query).reshape(-1)
            
            neighbor_feats, neighbor_mask = indexer.search(ref_pts_flat, batch_idx)
            
            # Prepare for MultiheadAttention
            # query: (L, N, E) -> (1, M, C)
            q_flat = query.permute(1, 0, 2).reshape(-1, 1, query.shape[-1]) 
            k_flat = neighbor_feats 
            v_flat = neighbor_feats
            
            key_padding_mask_lidar = ~neighbor_mask
            
            lidar_out, _ = self.lidar_cross_attn(
                q_flat, 
                k_flat, 
                v_flat, 
                key_padding_mask=key_padding_mask_lidar
            )
            
            lidar_out = lidar_out.reshape(bs, num_query, -1).permute(1, 0, 2)
            query_lidar = self.lidar_dropout(lidar_out)
        else:
            query_lidar = torch.zeros_like(query)

        # Coefficient Fusion
        fusion_weights = self.fusion_net(query) 
        alpha_cam = fusion_weights[..., 0:1]
        alpha_lidar = fusion_weights[..., 1:2]
        
        # Apply fusion
        query_fused = alpha_cam * query_cam + alpha_lidar * query_lidar
        
        # Norm
        query = self.norms[1](query_fused)
        
        # 3. FFN (Standard)
        query = self.ffns[0](query)
        query = self.norms[2](query)
        
        return query

@TRANSFORMER.register_module()
class UniADPTv3Transformer(PerceptionTransformer):
    def get_states_and_refs(
        self,
        bev_embed,
        object_query_embed,
        bev_h,
        bev_w,
        reference_points,
        reg_branches=None,
        cls_branches=None,
        img_metas=None,
        indexer=None 
    ):
        bs = bev_embed.shape[1]
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        
        reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.sigmoid()
        
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        # bev_embed = bev_embed.permute(1, 0, 2)

        spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=query.device, dtype=torch.long)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            key_padding_mask=None, 
            reference_points=reference_points, # Pass 3D reference points to Decoder
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=None,
            reg_branches=reg_branches,
            img_metas=img_metas,
            indexer=indexer
        )
        
        return inter_states, init_reference_out, inter_references
