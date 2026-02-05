
print("DEBUG: Script started")
import sys
import os
try:
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'projects'))
    
    import torch
    import numpy as np
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    print("DEBUG: Imports done")
except Exception as e:
    print(f"DEBUG: Import failed: {e}")
    sys.exit(1)

def main():
    print("Initializing...")
    config_file = 'projects/configs/stage1_track_map/base_track_map_lidar.py'
    
    try:
        cfg = Config.fromfile(config_file)
        print("Config loaded")
    except Exception as e:
        print(f"Config load failed: {e}")
        return

    print("Building model...")
    try:
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.cuda()
        model.train() # Set to train mode
        print(f"Model built: {type(model)}")
    except Exception as e:
        print(f"Model build failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check forward_train existence
    if hasattr(model, 'forward_train'):
        print("forward_train method found.")
    else:
        print("ERROR: forward_train method NOT found.")
        return

    # Mock Data for forward_train
    print("Preparing mock data...")
    batch_size = 1
    H, W = 256, 704
    T = 2
    img = torch.randn(batch_size, T, 6, 3, H, W).cuda()
    
    points = []
    for _ in range(batch_size):
        xyz = (torch.rand(2000, 3) - 0.5) * 100
        xyz[:, 2] = (torch.rand(2000) - 0.5) * 8
        feat = torch.zeros(2000, 2)
        pts = torch.cat([xyz, feat], dim=1).cuda()
        points.append(pts)

    # Note: img_metas needs to contain 'l2g_r_mat' and 'l2g_t' as numpy arrays
    # because BEVFormer's get_bev_features accesses them from img_metas
    meta = {
        'filename': ['mock.jpg']*6,
        'ori_shape': (900, 1600, 3),
        'img_shape': [(H, W, 3)]*6,
        'lidar2img': [np.eye(4)]*6, 
        'pad_shape': (H, W, 3),
        'scale_factor': 1.0,
        'flip': False,
        'pcd_horizontal_flip': False,
        'pcd_vertical_flip': False,
        'box_mode_3d': LiDARInstance3DBoxes,
        'box_type_3d': LiDARInstance3DBoxes,
        'img_norm_cfg': {'mean': [0,0,0], 'std': [1,1,1], 'to_rgb': False},
        'scene_token': 'test_token',
        'can_bus': np.zeros(18),
        'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        'sample_idx': 0,
        'l2g_r_mat': np.eye(3).astype(np.float32), # Added
        'l2g_t': np.zeros(3).astype(np.float32),   # Added
        'timestamp': 1613658236.123
    }
    
    img_metas_batch = []
    for _ in range(batch_size):
        frame_metas = []
        for t in range(T):
            m = meta.copy()
            m['scene_token'] = 'scene_1'
            m['timestamp'] = m['timestamp'] + t * 0.5
            frame_metas.append(m)
        img_metas_batch.append(frame_metas)
        
    # GT Data
    gt_bboxes_3d = []
    gt_labels_3d = []
    gt_inds = []
    gt_past_traj = []
    gt_past_traj_mask = []
    gt_sdc_bbox = []
    gt_sdc_label = []
    l2g_t = []
    l2g_r_mat = []
    timestamp = []
    
    for _ in range(batch_size):
        b_bboxes = []
        b_labels = []
        b_inds = []
        b_past_traj = []
        b_past_traj_mask = []
        b_sdc_bbox = []
        b_sdc_label = []
        b_l2g_t = []
        b_l2g_r_mat = []
        b_timestamp = []
        
        for t in range(T):
            # 1 box
            # Use 9-dim boxes (x, y, z, x_size, y_size, z_size, yaw, vx, vy) to match code_weights (10 dims after normalization)
            box = LiDARInstance3DBoxes(torch.tensor([[10, 10, 0, 4, 2, 1, 0, 0, 0]], dtype=torch.float).cuda(), box_dim=9)
            b_bboxes.append(box)
            b_labels.append(torch.tensor([0], dtype=torch.long).cuda())
            b_inds.append(torch.tensor([0], dtype=torch.long).cuda())
            # past_steps (4) + fut_steps (4) = 8 steps
            b_past_traj.append(torch.zeros(1, 8, 2).cuda())
            b_past_traj_mask.append(torch.zeros(1, 8).cuda())
            
            # SDC box also needs to be 9-dim
            sdc_box = LiDARInstance3DBoxes(torch.tensor([[0, 0, 0, 4, 2, 1, 0, 0, 0]], dtype=torch.float).cuda(), box_dim=9)
            b_sdc_bbox.append(sdc_box)
            b_sdc_label.append(torch.tensor([0], dtype=torch.long).cuda())
            
            b_l2g_t.append(torch.zeros(3).cuda())
            b_l2g_r_mat.append(torch.eye(3).cuda())
            b_timestamp.append(float(t)*0.5)
            
        gt_bboxes_3d.append(b_bboxes)
        gt_labels_3d.append(b_labels)
        gt_inds.append(b_inds)
        gt_past_traj.append(b_past_traj)
        gt_past_traj_mask.append(b_past_traj_mask)
        gt_sdc_bbox.append(b_sdc_bbox)
        gt_sdc_label.append(b_sdc_label)
        l2g_t.append(b_l2g_t)
        l2g_r_mat.append(b_l2g_r_mat)
        timestamp.append(b_timestamp)

    # Mock Map Data (Segmentation Head)
    gt_lane_labels = []
    gt_lane_bboxes = []
    gt_lane_masks = []
    
    for _ in range(batch_size):
         # Assuming 1 frame for map loss or it takes the last frame
         # PansegformerHead usually expects list of tensors
         # For verify script, let's pass list of tensors directly (simulating sliced data or queue_length=1 flattened)
         gt_lane_labels.append(torch.tensor([1]).long().cuda())
         gt_lane_bboxes.append(torch.tensor([[0,0,10,10]]).float().cuda())
         gt_lane_masks.append(torch.zeros((1, 200, 200)).float().cuda())

    print("Testing forward_train...")
    try:
        losses = model.forward_train(
            img=img,
            img_metas=img_metas_batch,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            l2g_t=l2g_t,
            l2g_r_mat=l2g_r_mat,
            timestamp=timestamp,
            points=points,
            gt_lane_labels=gt_lane_labels,
            gt_lane_bboxes=gt_lane_bboxes,
            gt_lane_masks=gt_lane_masks
        )
        print("forward_train executed successfully!")
        print("Loss keys:", losses.keys())
    except Exception as e:
        print(f"forward_train failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Verification Passed.")

if __name__ == "__main__":
    main()
