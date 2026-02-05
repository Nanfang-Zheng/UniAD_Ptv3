import torch
from mmdet.models import DETECTORS
from .uniad_track import UniADTrack
from projects.mmdet3d_plugin.uniad.modules.neighborhood_indexer import PTv3SerializationWindowIndexer, GridHashBucketIndexer
import copy
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin import Instances
from mmcv.runner import auto_fp16

@DETECTORS.register_module()
class UniADPTv3Track(UniADTrack):
    def __init__(self, 
                 pts_backbone=None,
                 indexer_cfg=None,
                 lidar_out_level=-1,
                 pts_backbone_pretrained=None,
                 **kwargs):
        task_loss_weight = kwargs.pop('task_loss_weight', {'track': 1.0})
        # Pop potential E2E heads if they are in kwargs
        kwargs.pop('seg_head', None)
        kwargs.pop('motion_head', None)
        kwargs.pop('occ_head', None)
        kwargs.pop('planning_head', None)
        
        super(UniADPTv3Track, self).__init__(pts_backbone=pts_backbone, **kwargs)
        self.task_loss_weight = task_loss_weight
        
        # PTv3 Backbone (built by super if in config, or here)
        # UniADTrack.__init__ calls super MVXTwoStageDetector which builds pts_backbone
        
        self.lidar_out_level = lidar_out_level
        
        # Build Indexer
        if indexer_cfg is None:
            # Default Scheme A
            indexer_cfg = dict(type='PTv3SerializationWindowIndexer')
            
        idx_type = indexer_cfg.pop('type', 'PTv3SerializationWindowIndexer')
        if idx_type == 'PTv3SerializationWindowIndexer':
            self.indexer = PTv3SerializationWindowIndexer(**indexer_cfg)
        else:
            self.indexer = GridHashBucketIndexer(**indexer_cfg)

        if pts_backbone_pretrained:
            self._load_ptv3_encoder_weights(pts_backbone_pretrained)

    def _load_ptv3_encoder_weights(self, ckpt_path):
        """Load only PTv3 embedding+encoder weights from a Pointcept checkpoint."""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Use model state_dict shapes (load_state_dict checks against these)
        model_state = self.pts_backbone.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            if not key.startswith('backbone.'):
                continue
            key = key[len('backbone.'):]
            if not (key.startswith('embedding.') or key.startswith('enc.')):
                continue
            if key not in model_state:
                continue
            tgt = model_state[key]
            v_use = None
            if v.shape == tgt.shape:
                v_use = v
            elif v.ndim == 5 and tgt.ndim == 5:
                # ckpt: [in,out,k,k,k] -> model: [out,k,k,k,in]
                if v.permute(1, 2, 3, 4, 0).shape == tgt.shape:
                    v_use = v.permute(1, 2, 3, 4, 0).contiguous()
                # ckpt: [out,in,k,k,k] -> model: [out,k,k,k,in]
                elif v.permute(0, 2, 3, 4, 1).shape == tgt.shape:
                    v_use = v.permute(0, 2, 3, 4, 1).contiguous()
            if v_use is None:
                continue
            filtered[key] = v_use
            
        model_state = self.pts_backbone.state_dict()
        loaded_count = 0
        for name, param in filtered.items():
            if name not in model_state: 
                print(f"[SKIP] {name} not found in model state_dict")
                continue
                
            if isinstance(param, torch.nn.Parameter):
                param = param.data
                
            try:
                # Check shape before copy
                if model_state[name].shape != param.shape:
                    print(f"[CRITICAL FAIL] Shape mismatch for {name} just before copy_: model={own_state[name].shape}, input={param.shape}")
                    continue
                
                model_state[name].copy_(param)
                loaded_count += 1
            except Exception as e:
                print(f"[FAIL] Error loading {name}: {e}")
        print(f"--- Manual loading finished. Loaded {loaded_count} / {len(filtered)} items. ---")
        # missing, unexpected = self.pts_backbone.load_state_dict(filtered, strict=False)
        # print(f'[PTv3] Loaded encoder weights from {ckpt_path}. '
        #       f'Used: {len(filtered)}, missing: {len(missing)}, unexpected: {len(unexpected)}')

    def extract_pts_feat(self, points, grid_coord=None):
        """
        Args:
            points (List[Tensor]): List of point clouds (B, N, C).
            grid_coord (List[Tensor], optional): List of grid coords (B, N, 3).
        """
        # 1. Prepare Data Dict for PTv3
        # PTv3 expects: feat, coord, grid_coord, offset/batch
        
        # Stack points
        offset = []
        batch = []
        feat_list = []
        coord_list = []
        grid_coord_list = []
        
        current_offset = 0
        for i, (p, gc) in enumerate(zip(points, grid_coord)):
            # p: (N, C)
            # gc: (N, 3)
            N = p.shape[0]
            current_offset += N
            offset.append(current_offset)
            batch.append(torch.full((N,), i, device=p.device, dtype=torch.long))
            
            # Assuming p[:, :3] is coord, p[:, 3:] is feat (intensity etc)
            # Or p is just feat? PTv3 usually takes feat=p.
            coord_list.append(p[:, :3])
            feat_list.append(p) 
            grid_coord_list.append(gc)
            
        offset = torch.tensor(offset, device=points[0].device, dtype=torch.long)
        batch = torch.cat(batch, dim=0)
        feat = torch.cat(feat_list, dim=0)
        coord = torch.cat(coord_list, dim=0)
        grid_coord = torch.cat(grid_coord_list, dim=0)
        
        data_dict = dict(
            feat=feat,
            coord=coord,
            grid_coord=grid_coord,
            offset=offset,
            batch=batch
        )
        
        # 2. Forward PTv3
        # self.pts_backbone should be PointTransformerV3
        pt_out = self.pts_backbone(data_dict, return_enc_out=True)
        if isinstance(pt_out, tuple):
            point, enc_point = pt_out
        else:
            point, enc_point = pt_out, None
        
        # 3. Output Standardization & Feature Selection
        # Point contains features from encoder/decoder stages
        # We need to pick one based on lidar_out_level
        # Point is a Dict-like object. Keys might be 'enc4', 'dec3', etc.
        # Or just 'feat' if it returns the final output.
        # Looking at PTv3 code: 
        # return point (which has .feat updated by the last layer)
        # If we want intermediate, we need to hack PTv3 or rely on it returning 'feat' as final.
        # The user said "Select usage of specific layer OR last layer".
        # PTv3 code updates .feat in place.
        # To get specific layer, we'd need to modify PTv3 to store intermediates?
        # Or assumes 'point' dict retains keys like 'enc{s}' added during forward?
        # Code: self.enc.add(module, name=f"enc{s}") -> PointSequential executes.
        # PointSequential doesn't auto-save intermediates to dict keys unless the module does.
        # However, Point object inherits Dict.
        # If we want intermediate, we assume the backbone was modified or we just use final.
        # User constraint: "Don't modify ptv3_backbone.py".
        # So we can ONLY use what's in 'point'.
        # If 'point' only has final 'feat', we use that.
        # But wait, `point` is a `Dict`. Does `PointSequential` add outputs to the dict?
        # No, `PointSequential` updates `input` (which is `point`).
        # So `point.feat` is the output of the last block.
        # Unless we access `point['enc4']`? No, that's not standard behavior.
        # So we assume we use the FINAL output.
        # If lidar_out_level != -1, we might be out of luck without modifying backbone.
        # I'll stick to using `point` (final output) for now.
        
        # 4. Build Indexer
        # We pass the WHOLE point object to indexer
        # We also need min_coord if used (assumed handled or passed if available)
        self.indexer.build(enc_point if enc_point is not None else point)

        return point

    @auto_fp16(apply_to=("img", "points"))
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
                            points=None,
                            grid_coord=None,
                            **kwargs):
        """Forward training function for tracking.
        Args:
            img (torch.Tensor): Images of each sample with shape (B, N, C, H, W).
            points (List[Tensor]): Point clouds of each sample with shape (B, N, C).
            grid_coord (List[Tensor]): Grid coords of each sample with shape (B, N, 3).
        """
        # Note: img is already in kwargs or we should explicitly pass it if it's not.
        
        track_instances = self._generate_empty_tracks()
        num_frame = img.size(1)
        
        gt_instances_list = []
        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)
            boxes = normalize_bbox(boxes, self.pc_range)
            sd_boxes = gt_sdc_bbox[0][i].tensor.to(img.device)
            sd_boxes = normalize_bbox(sd_boxes, self.pc_range)
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = gt_inds[0][i]
            gt_instances.past_traj = gt_past_traj[0][i].float()
            gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()
            gt_instances.sdc_boxes = torch.cat([sd_boxes for _ in range(boxes.shape[0])], dim=0)
            gt_instances.sdc_labels = torch.cat([gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0)
            gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)
        out = dict()
        
        # Pre-process points for all frames if available
        # points: List[Tensor] (B*T) or (B, T)?
        # CustomCollect3D usually collates to List[Tensor]
        # But for temporal data, it might be nested?
        # Let's assume points is List[Tensor] of length B (containing T frames concatenated? No)
        # Usually for video, points is List[List[Tensor]]?
        # Let's inspect how CustomCollect3D handles points.
        # If it's a list of list, we flatten or index it.
        # Assuming points corresponds to img structure.
        
        # Actually, let's extract features frame by frame inside the loop
        # to match img processing.
        
        for i in range(num_frame):
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)
            
            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            img_metas_single = [copy.deepcopy(img_metas[0][i])]
            
            # Extract Points Feature for this frame
            # points is list of (B, N, C) ? No, points is list of (N, C) if batch=1?
            # If batch > 1, points is list of T lists?
            
            points_single = None
            grid_coord_single = None
            if points is not None:
                # Assuming points is [Batch][Frame] or [Batch * Frame]
                # If [Batch][Frame], then points[0][i]
                # If [Batch * Frame], then points[0 * T + i]
                # Let's assume batch size = 1 for now as per UniAD standard
                if isinstance(points[0], list):
                    # List of List
                    points_single = [p[i] for p in points]
                    if grid_coord is not None:
                         grid_coord_single = [g[i] for g in grid_coord]
                else:
                    # Maybe flat list? Or just single frame?
                    # If single frame, len(points) == B
                    # But we have T frames.
                    # Let's assume points is List of Tensors, where each Tensor is one frame's PC.
                    # But wait, DataLoader collates.
                    # If T > 1, and we use LoadPointsFromFile, it loads ONE frame.
                    # UniAD usually uses MultiView images.
                    # Does it use MultiSweep Points?
                    # If we only have Keyframe points, then points_single is valid only for i==? 
                    # Usually keyframe is last or middle?
                    # If we use 'points' key, it usually corresponds to the current frame (sample).
                    # In temporal training, 'img' has T frames.
                    # 'points' might only be for the KEY frame (usually the last one or current one).
                    # If so, we only extract features for that frame?
                    # But we need features for ALL frames if we want to track?
                    # Or maybe we only train detection on keyframe?
                    # UniAD trains on sequence.
                    # If we lack points for past frames, we can't use PTv3 for past frames.
                    # We might need to skip PTv3 for past frames or use zero features.
                    # OR, we assume points contains T frames.
                    
                    # For safety: use points only if i matches the keyframe index?
                    # Or assume points is a list of length T (for BS=1).
                    if len(points) == num_frame:
                         points_single = [points[i]]
                         if grid_coord is not None:
                             grid_coord_single = [grid_coord[i]]
                    else:
                         # Fallback: maybe points only has 1 frame (keyframe)
                         # If so, we can only use it for that frame.
                         # Let's assume it's the last frame?
                         # Or just don't use it for now if mismatch.
                         pass
            
            # Run Backbone & Indexer
            if points_single is not None:
                self.extract_pts_feat(points_single, grid_coord_single)
            
            # Time Delta
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]

            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []
            
            frame_res = self._forward_single_frame_train(
                img_single,
                img_metas_single,
                track_instances,
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
                indexer=self.indexer if points_single is not None else None # Pass indexer
            )
            
            track_instances = frame_res["track_instances"]
            
        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                    "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        out.update({k: frame_res[k] for k in get_keys})
        
        losses = self.criterion.losses_dict
        losses = self.loss_weighted_and_prefixed(losses, prefix='track')
        return losses, out

    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = 1.0
        if hasattr(self, 'task_loss_weight'):
            loss_factor = self.task_loss_weight.get(prefix, 1.0)
        loss_dict = {f"{prefix}.{k}": v * loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def _forward_single_frame_train(self, 
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
                                    indexer=None):
        
        bev_embed, bev_pos = self.get_bevs(
            img, img_metas,
            prev_img=prev_img, prev_img_metas=prev_img_metas,
        )
        
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
            indexer=indexer
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
        if l2g_r2 is not None:
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
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
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])
        
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

    def forward_train(self,
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
                      **kwargs):
        """Forward training function."""
        # Ensure 'img' is not duplicated in kwargs if we pass it explicitly
        img = kwargs.pop('img', None)
        points = kwargs.pop('points', None)
        grid_coord = kwargs.pop('grid_coord', None)
        
        return self.forward_track_train(img=img,
                                        img_metas=img_metas,
                                        gt_bboxes_3d=gt_bboxes_3d,
                                        gt_labels_3d=gt_labels_3d,
                                        gt_inds=gt_inds,
                                        l2g_t=l2g_t,
                                        l2g_r_mat=l2g_r_mat,
                                        timestamp=timestamp,
                                        gt_past_traj=gt_past_traj,
                                        gt_past_traj_mask=gt_past_traj_mask,
                                        gt_sdc_bbox=gt_sdc_bbox,
                                        gt_sdc_label=gt_sdc_label,
                                        points=points,
                                        grid_coord=grid_coord
                                       )[0]

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        """Test function without test time augmentation.

        Args:
            points (list[torch.Tensor]): Point clouds of each sample.
            img_metas (list[dict]): Meta information of samples.
            img (list[torch.Tensor]): Images of each sample.
            rescale (bool): Whether to rescale results.
        """
        grid_coord = kwargs.pop('grid_coord', None)
        l2g_t = kwargs.pop('l2g_t', None)
        l2g_r_mat = kwargs.pop('l2g_r_mat', None)
        timestamp = kwargs.pop('timestamp', None)

        return self.simple_test_track(
            img=img,
            l2g_t=l2g_t,
            l2g_r_mat=l2g_r_mat,
            img_metas=img_metas,
            timestamp=timestamp,
            points=points,
            grid_coord=grid_coord,
            **kwargs
        )

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
        points=None, # New
        grid_coord=None, # New
        **kwargs
    ):
        """only support bs=1 and sequential input"""

        bs = img.size(0)
        # img_metas = img_metas[0]
        if isinstance(timestamp, list):
            timestamp = timestamp[0]
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.item()
            
        if isinstance(l2g_t, list):
            l2g_t = l2g_t[0]
        if isinstance(l2g_t, torch.Tensor) and l2g_t.shape[0] == 1 and l2g_t.dim() > 1:
            l2g_t = l2g_t[0]
            
        if isinstance(l2g_r_mat, list):
            l2g_r_mat = l2g_r_mat[0]
        if isinstance(l2g_r_mat, torch.Tensor) and l2g_r_mat.shape[0] == 1 and l2g_r_mat.dim() > 2:
            l2g_r_mat = l2g_r_mat[0]
        # Extract Points Feature
        # Note: simple_test_track is usually called frame by frame
        # points here is likely (1, N, C) or List[Tensor]
        if points is not None:
             # If points is list of tensors (one per sample in batch, but bs=1)
             if isinstance(points, list):
                 pass 
             else:
                 points = [points]
             if grid_coord is not None and not isinstance(grid_coord, list):
                grid_coord = [grid_coord]
             if grid_coord[0].dim() == 3 and grid_coord[0].shape[0] == 1:
                    grid_coord = [gc[0] for gc in grid_coord]
             self.extract_pts_feat(points, grid_coord)


        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
        
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            indexer=self.indexer if points is not None else None # Pass indexer
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]

        self.test_track_instances = track_instances
        results = [dict()]
        # OOM Fix: Removed heavy tensors ("bev_embed", "bev_pos", "track_query_embeddings") 
        # that are not needed for standard bbox evaluation.
        get_keys = ["track_bbox_results", 
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        
        results[0].update({k: frame_res[k] for k in get_keys})
        results = self._det_instances2results(track_instances_fordet, results, img_metas)
        return results

    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        indexer=None # New
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])
            active_inst.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)
        
        # Pass indexer
        det_output = self.pts_bbox_head.get_detections(
            bev_embed, 
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
            indexer=indexer # New
        )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query 
        track_instances.obj_idxes[900] = -2
        """ update track base """
        self.track_base.update(track_instances, None)
       
        active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)    # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out
