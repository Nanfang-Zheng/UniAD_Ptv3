#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.models.builder import (
    build_backbone, 
    build_neck, 
    build_voxel_encoder, 
    build_middle_encoder,
    build_head
)
from mmdet3d.ops import Voxelization
from mmcv.runner import auto_fp16
from .uniad_track import UniADTrack
from .utils import pop_elem_in_result
import copy

@DETECTORS.register_module()
class UniADLidarTrack(UniADTrack):
    """
    UniAD with LiDAR-Vision Fusion for Tracking.
    Implements Cross-Attention based fusion in the detector.
    """
    
    def __init__(self, 
                 pts_voxel_layer=None, 
                 pts_voxel_encoder=None, 
                 pts_middle_encoder=None, 
                 pts_backbone=None, 
                 pts_neck=None,
                 fusion_cfg=dict(
                     embed_dim=256,
                     num_heads=4,
                     lidar_dim=384, 
                     dropout=0.1
                 ),
                 # Arguments from config that UniADTrack might not handle but are present in config
                 seg_head=None,
                 motion_head=None,
                 occ_head=None,
                 planning_head=None,
                 task_loss_weight=dict(
                    track=1.0,
                    map=1.0,
                    motion=1.0,
                    occ=1.0,
                    planning=1.0
                 ),
                 pretrained=None,
                 **kwargs):
        # Initialize UniADTrack
        super(UniADLidarTrack, self).__init__(pretrained=pretrained, **kwargs)
        
        # Initialize Heads (Similar to UniAD)
        if seg_head:
            self.seg_head = build_head(seg_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if motion_head:
            self.motion_head = build_head(motion_head)
        if planning_head:
            self.planning_head = build_head(planning_head)
        
        self.task_loss_weight = task_loss_weight
        
        # Initialize LiDAR components
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = build_middle_encoder(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone)
            if pretrained and isinstance(pretrained, dict) and pretrained.get('pts_backbone_path', None):
                self.pts_backbone.init_weights(pretrained=pretrained['pts_backbone_path'])
        if pts_neck:
            self.pts_neck = build_neck(pts_neck)
            
        # Initialize Fusion Components (Cross-Attention)
        self.fusion_dim = fusion_cfg.get('embed_dim', 256)
        self.lidar_dim = fusion_cfg.get('lidar_dim', 384) # Default SECONDFPN output
        
        # Project LiDAR features to match embed_dim
        self.lidar_input_proj = nn.Conv2d(self.lidar_dim, self.fusion_dim, kernel_size=1)
        
        # Cross Attention: Query=Image, Key/Value=LiDAR
        # Use Downsampled Attention to save memory (200x200 -> 50x50)
        self.downsample_factor = 4
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim, 
            num_heads=fusion_cfg.get('num_heads', 4),
            dropout=fusion_cfg.get('dropout', 0.1),
            batch_first=False 
        )
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None
    
    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      
                      # Occ_gt
                      gt_segmentation=None,
                      gt_instance=None, 
                      gt_occ_img_is_valid=None,
                      
                      #planning
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      
                      # fut gt for planning
                      gt_future_boxes=None,
                      **kwargs,  # [1, 9]
                      ):
        """Forward training function."""
        losses = dict()
        len_queue = img.size(1)
        
        # NOTE: Pass points explicitly if they are in kwargs
        points = kwargs.get('points', None)

        losses_track, outs_track = self.forward_track_train(img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp, points=points)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)
        
        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)

        bev_embed = outs_track["bev_embed"]
        bev_pos  = outs_track["bev_pos"]

        img_metas = [each[len_queue-1] for each in img_metas]

        outs_seg = dict()
        if self.with_seg_head:          
            losses_seg, outs_seg = self.seg_head.forward_train(bev_embed, img_metas,
                                                          gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
            
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            losses.update(losses_seg)

        outs_motion = dict()
        # Forward Motion Head
        if self.with_motion_head:
            ret_dict_motion = self.motion_head.forward_train(bev_embed,
                                                        gt_bboxes_3d, gt_labels_3d, 
                                                        gt_fut_traj, gt_fut_traj_mask, 
                                                        gt_sdc_fut_traj, gt_sdc_fut_traj_mask, 
                                                        outs_track=outs_track, outs_seg=outs_seg
                                                    )
            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion['bev_pos'] = bev_pos
            losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
            losses.update(losses_motion)

        # Forward Occ Head
        if self.with_occ_head:
            if outs_motion['track_query'].shape[1] == 0:
                # TODO: rm hard code
                outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['track_query_pos'] = torch.zeros((1,1, 256)).to(bev_embed)
                outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion['all_matched_idxes'] = [[-1]]
            losses_occ = self.occ_head.forward_train(
                            bev_embed,
                            outs_motion,
                            gt_inds_list=gt_inds,
                            gt_segmentation=gt_segmentation,  
                            gt_instance=gt_instance, 
                            gt_img_is_valid=gt_occ_img_is_valid,
                        )
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
            losses.update(losses_occ)
        

        # Forward Plan Head
        if self.with_planning_head:
            outs_planning = self.planning_head.forward_train(bev_embed, outs_motion, sdc_planning, sdc_planning_mask, command, gt_future_boxes)
            losses_planning = outs_planning['losses']
            losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)
        
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)
        return losses
    
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     # planning gt(for evaluation only)
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
 
                     # Occ_gt (for evaluation only)
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,
                     **kwargs
                    ):
        """Test function"""
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # first frame
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # following frames
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)

        # Upsample bev for tiny model        
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]

        if self.with_seg_head:
            result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
            outs_motion['bev_pos'] = result_track[0]['bev_pos']

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed, 
                outs_motion,
                no_query = occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ
        
        if self.with_planning_head:
            planning_gt=dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
            result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command)
            result[0]['planning'] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]['occ'] = pop_elem_in_result(result[0]['occ'],  \
                pop_list=['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid'])
        
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result

    @torch.no_grad()
    def extract_pts_feat(self, points):
        """Extract LiDAR BEV features"""
        if not hasattr(self, 'pts_voxel_layer'):
            return None
            
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
            
        # SECONDFPN returns a tuple/list, usually we want the concatenated feature
        # If it returns a tuple, take the last one or concat them?
        # Standard SECONDFPN in mmdet3d returns a tuple/list of features.
        # But if upsample_strides are set, it might return a single concatenated tensor 
        # or we need to concat manually. 
        # Let's check if it is a list/tuple.
        if isinstance(x, (list, tuple)):
            # Assuming they are already upsampled and same size
            return torch.cat(x, dim=1)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
        
        Args:
            points (list[torch.Tensor]): Points of each sample.
        
        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None, points=None):
        """
        Override get_bevs to include LiDAR fusion.
        """
        # 1. Get Image BEV (Query)
        # Call parent's get_bevs
        # Note: UniADTrack.get_bevs returns (bev_embed, bev_pos)
        # bev_embed shape: (L, B, C) where L = H*W
        img_bev, bev_pos = super().get_bevs(imgs, img_metas, prev_img, prev_img_metas, prev_bev)
        
        # 2. Fuse with LiDAR BEV if points are provided
        if points is not None and hasattr(self, 'pts_backbone'):
            lidar_bev = self.extract_pts_feat(points) # (B, C_lidar, H_l, W_l)
            
            if lidar_bev is not None:
                # Align Dimensions
                # img_bev is (L, B, C). We need to know H, W.
                # self.bev_h, self.bev_w are available in UniADTrack
                
                # Resize LiDAR BEV to match Image BEV spatial size
                lidar_bev = F.interpolate(lidar_bev, size=(self.bev_h, self.bev_w), mode='bilinear', align_corners=True)
                
                # Project Channel dimensions: (B, C_lidar, H, W) -> (B, C_embed, H, W)
                lidar_bev = self.lidar_input_proj(lidar_bev)
                
                # Flatten: (B, C, H, W) -> (B, C, L) -> (B, L, C) -> (L, B, C)
                lidar_bev_flat = lidar_bev.flatten(2).permute(2, 0, 1)
                
                # Positional Embedding
                # bev_pos is (B, C, H, W) in head, but get_bevs returns what?
                # UniADTrack.get_bevs: bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(...)
                # BEVFormerTrackHead.get_bev_features returns bev_pos as (B, C, H, W).
                # Wait, UniADTrack.get_bevs does NOT permute bev_pos.
                
                # Flatten bev_pos for attention
                pos_flat = bev_pos.flatten(2).permute(2, 0, 1) # (L, B, C)
                
                # Cross Attention Fusion with Downsampling to avoid OOM
                # img_bev: (L, B, C) -> (B, C, H, W)
                L, B, C = img_bev.shape
                H, W = self.bev_h, self.bev_w
                
                img_bev_spatial = img_bev.permute(1, 2, 0).view(B, C, H, W)
                lidar_bev_spatial = lidar_bev # (B, C, H, W)
                pos_spatial = bev_pos # (B, C, H, W)
                
                # Downsample
                img_bev_small = F.adaptive_avg_pool2d(img_bev_spatial, (H//self.downsample_factor, W//self.downsample_factor))
                lidar_bev_small = F.adaptive_avg_pool2d(lidar_bev_spatial, (H//self.downsample_factor, W//self.downsample_factor))
                pos_small = F.adaptive_avg_pool2d(pos_spatial, (H//self.downsample_factor, W//self.downsample_factor))
                
                # Flatten
                img_bev_small_flat = img_bev_small.flatten(2).permute(2, 0, 1)
                lidar_bev_small_flat = lidar_bev_small.flatten(2).permute(2, 0, 1)
                pos_small_flat = pos_small.flatten(2).permute(2, 0, 1)
                
                query = img_bev_small_flat + pos_small_flat
                key = lidar_bev_small_flat + pos_small_flat
                value = lidar_bev_small_flat
                
                # attn_out: (L_small, B, C)
                attn_out_small, _ = self.fusion_attn(query, key, value)
                
                # Upsample back to H, W
                attn_out_small_spatial = attn_out_small.permute(1, 2, 0).view(B, C, H//self.downsample_factor, W//self.downsample_factor)
                attn_out_spatial = F.interpolate(attn_out_small_spatial, size=(H, W), mode='bilinear', align_corners=True)
                
                # Flatten back
                attn_out = attn_out_spatial.flatten(2).permute(2, 0, 1)
                
                # Residual Connection + Norm
                # Fused = Norm(Image + Attention)
                img_bev = self.fusion_norm(img_bev + attn_out)
                
        return img_bev, bev_pos

    @auto_fp16(apply_to=('img', 'points'))
    def forward_track_train(self,
                            img,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            gt_past_traj,
                            gt_past_traj_mask,
                            gt_inds,
                            gt_sdc_bbox,
                            gt_sdc_label,
                            l2g_t,
                            l2g_r_mat,
                            img_metas,
                            timestamp,
                            points=None):
        """
        Override to pass points to internal methods.
        We need to copy most of the logic because get_bevs is called inside _forward_single_frame_train
        and we need to pass points down the stack.
        """
        # Because _forward_single_frame_train calls get_bevs, and get_bevs needs points,
        # we need to override _forward_single_frame_train as well OR monkey-patch it.
        # Overriding is cleaner but involves copying code.
        # Let's override _forward_single_frame_train in this class.
        
        # Store points temporarily to access in _forward_single_frame_train
        # This avoids changing the signature of _forward_single_frame_train which might be used elsewhere
        self.temp_points = points
        
        return super().forward_track_train(
            img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, 
            gt_inds, gt_sdc_bbox, gt_sdc_label, l2g_t, l2g_r_mat, img_metas, timestamp
        )

    @auto_fp16(apply_to=("img", "prev_bev"))
    def _forward_single_frame_train(
        self,
        img,
        img_metas,
        track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
    ):
        """
        Override to pass points to get_bevs
        """
        # Retrieve points
        points = getattr(self, 'temp_points', None)
        
        # NOTE: We assume batch size 1 for now as per original code warning
        # points should be a list of tensors for BS=1
        
        # We need to slice points for the current frame if it's a sequence?
        # forward_track_train receives points for the whole clip?
        # Actually UniAD input `points` is usually list[list[Tensor]] for (B, N_sweeps)?
        # No, typically points is list[Tensor] of size B.
        # But forward_track_train iterates over frames (num_frame = img.size(1)).
        # Does points have time dimension?
        # In LoadPointsFromFile, it's one frame.
        # If we are training with video, we might have multiple frames of points?
        # UniAD usually trains on keyframes.
        # Let's assume points corresponds to the current keyframe or we use the same points (not ideal for tracking).
        
        # Issue: The standard pipeline for UniAD video training might not load points for all frames in the queue?
        # In base_track_map_lidar.py:
        # train_pipeline has LoadPointsFromFile.
        # CustomCollect3D keys include 'points'.
        # If queue_length > 1, 'img' is (B, N_frames, ...).
        # 'points' might be (B, ) but usually Collect3D just gathers them.
        # If we use sequential training, we need points for each frame.
        
        # Simplification: For now, use the provided points for the current frame calculation.
        # Since we are modifying get_bevs, we just pass what we have.
        
        bev_embed, bev_pos = self.get_bevs(
            img, img_metas,
            prev_img=prev_img, prev_img_metas=prev_img_metas,
            points=points # Pass points here
        )
        
        # The rest is identical to parent.
        # We can call the parent's logic but we need to inject our bev_embed.
        # But get_bevs is called inside parent's _forward_single_frame_train.
        # So we HAVE to copy the whole method to replace get_bevs call.
        
        # ... Copying _forward_single_frame_train logic ...
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )
        
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }
        
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        nb_dec = output_classes.size(0)
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]
        velo = output_coords[-1, 0, :, -2:]
        
        # kwargs has l2g_r2 etc.
        # l2g_r2 = kwargs.get('l2g_r2')
        # l2g_t1 = kwargs.get('l2g_t1')
        # l2g_r1 = kwargs.get('l2g_r1')
        # l2g_t2 = kwargs.get('l2g_t2')
        # time_delta = kwargs.get('time_delta')
        
        if l2g_r2 is not None:
            ref_pts = self.velo_update(
                last_ref_pts[0], velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
        else:
            ref_pts = last_ref_pts[0]

        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
        track_instances.ref_pts[...,:2] = ref_pts[...,:2]
        
        track_instances_list.append(track_instances)
        
        for i in range(nb_dec):
            track_instances = track_instances_list[i]
            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]
            track_instances.pred_boxes = output_coords[i, 0]
            track_instances.pred_past_trajs = output_past_trajs[i, 0]
            out["track_instances"] = track_instances
            
            # These are lists passed in kwargs
            # all_query_embeddings = kwargs.get('all_query_embeddings')
            # all_matched_indices = kwargs.get('all_matched_indices')
            # all_instances_pred_logits = kwargs.get('all_instances_pred_logits')
            # all_instances_pred_boxes = kwargs.get('all_instances_pred_boxes')

            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            if all_query_embeddings is not None: all_query_embeddings.append(query_feats[i][0])
            if all_matched_indices is not None: all_matched_indices.append(matched_indices)
            if all_instances_pred_logits is not None: all_instances_pred_logits.append(output_classes[i, 0])
            if all_instances_pred_boxes is not None: all_instances_pred_boxes.append(output_coords[i, 0])
        
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[900], img_metas))
        
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        return out
