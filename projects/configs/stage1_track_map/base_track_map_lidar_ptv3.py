_base_ = ["./base_track_map.py"]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# Point Cloud Range
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# Occ args (Copied from base to avoid _base_ access error)
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# PTv3 Config
pts_backbone = dict(
    type="PointTransformerV3",
    in_channels=4, 
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    cls_mode=True,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
)

model = dict(
    type="UniADPTv3Track",
    pts_backbone=pts_backbone,
    pts_backbone_pretrained="/home/zhengnanfang/PTv3_proj/UniAD/ckpts/model_best.pth",
    lidar_out_level=-1,
    indexer_cfg=dict(
        type='PTv3SerializationWindowIndexer',
        window_size=128,
        depth=16,
        grid_size=0.05
    ),
    pts_bbox_head=dict(
        type="UniADPTv3TrackHead",
        transformer=dict(
            type="UniADPTv3Transformer",
            decoder=dict(
                type="UniADPTv3Decoder",
                transformerlayers=dict(
                    type="UniADPTv3DecoderLayer",
                    lidar_cross_attn_cfg=dict(
                        embed_dim=256,
                        num_heads=8,
                        dropout=0.1,
                        batch_first=True
                    ),
                    fusion_cfg=dict(
                        embed_dims=256
                    )
                )
            )
        )
    )
)

# Pipeline
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True, file_client_args=dict(backend="disk"), img_root=""),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D_E2E",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_future_anns=True,
        with_ins_inds_3d=True,
        ins_inds_add_1=True,
    ),
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, filter_invisible=False),
    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    
    # PTv3 Preprocessing
    dict(
        type="GridSample_migrate",
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_min_coord=True
    ),
    
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "gt_bboxes_3d", "gt_labels_3d", "gt_inds", "img", "points", "grid_coord",
            "timestamp", "l2g_r_mat", "l2g_t", "gt_fut_traj", "gt_fut_traj_mask",
            "gt_past_traj", "gt_past_traj_mask", "gt_sdc_bbox", "gt_sdc_label",
            "gt_sdc_fut_traj", "gt_sdc_fut_traj_mask", "gt_lane_labels", "gt_lane_bboxes", "gt_lane_masks",
            "gt_segmentation", "gt_instance", "gt_centerness", "gt_offset", "gt_flow", "gt_backward_flow",
            "gt_occ_has_invalid_frame", "gt_occ_img_is_valid", "gt_future_boxes", "gt_future_labels",
            "sdc_planning", "sdc_planning_mask", "command",
        ],
        meta_keys=[
            "filename", "ori_shape", "img_shape", "lidar2img", "depth2img", "cam2img", "pad_shape",
            "scale_factor", "flip", "pcd_horizontal_flip", "pcd_vertical_flip", "box_mode_3d",
            "box_type_3d", "img_norm_cfg", "pcd_trans", "sample_idx", "prev_idx", "next_idx",
            "pcd_scale_factor", "pcd_rotation", "pts_filename", "transformation_3d_flow", "scene_token",
            "can_bus","l2g_r_mat","min_coord" #添加了min_coord
        ]
    ),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True, file_client_args=dict(backend="disk"), img_root=""),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type='LoadAnnotations3D_E2E', with_bbox_3d=False, with_label_3d=False, with_attr_label=False, with_future_anns=True, with_ins_inds_3d=False, ins_inds_add_1=True),
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, filter_invisible=False),
    
    dict(
        type="GridSample_migrate",
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_min_coord=True
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(
                type="CustomCollect3D", 
                keys=[
                    "img", "points", "grid_coord",
                    "timestamp", "l2g_r_mat", "l2g_t", "gt_lane_labels", "gt_lane_bboxes", "gt_lane_masks",
                    "gt_segmentation", "gt_instance", "gt_centerness", "gt_offset", "gt_flow", "gt_backward_flow",
                    "gt_occ_has_invalid_frame", "gt_occ_img_is_valid", "sdc_planning", "sdc_planning_mask", "command",
                ],
                meta_keys=[
                    "filename", "ori_shape", "img_shape", "lidar2img", "depth2img", "cam2img", "pad_shape",
                    "scale_factor", "flip", "pcd_horizontal_flip", "pcd_vertical_flip", "box_mode_3d",
                    "box_type_3d", "img_norm_cfg", "pcd_trans", "sample_idx", "prev_idx", "next_idx",
                    "pcd_scale_factor", "pcd_rotation", "pts_filename", "transformation_3d_flow", "scene_token",
                    "can_bus","l2g_r_mat","min_coord"
                ]
            ),
        ],
    ),
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
