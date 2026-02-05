# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from mmcv.runner import force_fp32, auto_fp16
from nuscenes.utils.data_classes import Quaternion

@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        """初始化PerceptionTransformer类
        
        Args:
            num_feature_levels (int): FPN特征金字塔的特征级别数量，默认为4
            num_cams (int): 相机数量，默认为6
            two_stage_num_proposals (int): 两阶段检测时生成的提议数量，默认为300
            encoder (dict): 编码器配置字典，用于构建编码器层序列
            decoder (dict): 解码器配置字典，用于构建解码器层序列
            embed_dims (int): 嵌入维度，默认为256
            rotate_prev_bev (bool): 是否旋转之前的BEV特征以对齐当前帧，默认为True
            use_shift (bool): 是否使用位移信息来对齐BEV特征，默认为True
            use_can_bus (bool): 是否使用CAN总线信息，默认为True
            can_bus_norm (bool): 是否对CAN总线信息进行归一化，默认为True
            use_cams_embeds (bool): 是否为不同相机添加相机嵌入，默认为True
            rotate_center (list): 旋转BEV特征的中心坐标，默认为[100, 100]
            **kwargs: 其他传递给父类BaseModule的参数
        """
        super(PerceptionTransformer, self).__init__(**kwargs)
        # 构建编码器层序列
        self.encoder = build_transformer_layer_sequence(encoder)
        # 构建解码器层序列
        self.decoder = build_transformer_layer_sequence(decoder)
        # 设置嵌入维度
        self.embed_dims = embed_dims
        # 设置特征级别数量
        self.num_feature_levels = num_feature_levels
        # 设置相机数量
        self.num_cams = num_cams
        # 初始化fp16支持为False
        self.fp16_enabled = False

        # 保存旋转BEV特征的配置
        self.rotate_prev_bev = rotate_prev_bev
        # 保存位移使用配置
        self.use_shift = use_shift
        # 保存CAN总线使用配置
        self.use_can_bus = use_can_bus
        # 保存CAN总线归一化配置
        self.can_bus_norm = can_bus_norm
        # 保存相机嵌入使用配置
        self.use_cams_embeds = use_cams_embeds

        # 保存两阶段提议数量
        self.two_stage_num_proposals = two_stage_num_proposals
        # 初始化各层
        self.init_layers()
        # 保存旋转中心
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,  
            bev_queries,
            bev_h,
            bev_w,
            real_h,
            real_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            img_metas=None):
        """
        从多视角图像特征中获取鸟瞰图(BEV)特征
        
        Args:
            mlvl_feats (list[Tensor]): 多尺度图像特征列表，每个元素形状为[bs, num_cam, embed_dims, h, w]
            bev_queries (Tensor): BEV查询向量，形状为[bev_h*bev_w, embed_dims]
            bev_h (int): BEV特征图的高度
            bev_w (int): BEV特征图的宽度
            real_h (float): 真实世界中BEV区域的高度(m)
            real_w (float): 真实世界中BEV区域的宽度(m)
            grid_length (list[float], optional): BEV网格单元的实际尺寸(m)，默认为[0.512, 0.512]
            bev_pos (Tensor, optional): BEV位置编码，形状为[bs, embed_dims, bev_h, bev_w]，默认为None
            prev_bev (Tensor, optional): 前一帧的BEV特征，形状为[bev_h*bev_w, bs, embed_dims]，默认为None
            img_metas (list[dict], optional): 图像元数据列表，包含CAN总线信息等，默认为None
        
        Returns:
            Tensor: 生成的BEV特征，形状为[bev_h*bev_w, bs, embed_dims]
        """

        # 获取批次大小
        bs = mlvl_feats[0].size(0)
        # 扩展BEV查询向量到批次维度，形状变为[bev_h*bev_w, bs, embed_dims]
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        # 处理BEV位置编码，形状从[bs, embed_dims, bev_h, bev_w]变为[bev_h*bev_w, bs, embed_dims]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        
        # 从CAN总线获取自车运动增量(全局坐标系)
        delta_global = np.array([each['can_bus'][:3] for each in img_metas])
        # 获取激光雷达到全局坐标系的旋转矩阵
        lidar2global_rotation = np.array([each['l2g_r_mat'] for each in img_metas])
        # 将自车运动增量转换到激光雷达坐标系
        delta_lidar = []
        for i in range(bs): 
            delta_lidar.append(np.linalg.inv(lidar2global_rotation[i]) @ delta_global[i])
        delta_lidar = np.array(delta_lidar)
        
        # 计算BEV网格中的位移量
        shift_y = delta_lidar[:, 1] / real_h
        shift_x = delta_lidar[:, 0] / real_w
        # 根据配置决定是否使用位移信息
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        # 转换为张量并调整维度，形状为[bs, 2]
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)

        # 如果存在前一帧BEV特征，进行处理
        if prev_bev is not None:
            # 调整前一帧BEV特征的维度
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            # 根据配置决定是否旋转前一帧BEV特征以对齐当前帧
            if self.rotate_prev_bev:
                for i in range(bs):
                    # 获取旋转角度
                    rotation_angle = img_metas[i]['can_bus'][-1]
                    # 调整前一帧BEV特征的维度以便旋转
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    # 旋转BEV特征
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    # 恢复原维度
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # 添加CAN总线信号到BEV查询
        can_bus = bev_queries.new_tensor([each['can_bus'] for each in img_metas])  # 形状为[bs, 18]
        # 通过MLP处理CAN总线信号并扩展维度
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]  # 形状为[1, bs, embed_dims]
        # 根据配置决定是否添加CAN总线信息
        bev_queries = bev_queries + can_bus * self.use_can_bus

        # 处理多尺度图像特征
        feat_flatten = []  # 存储扁平化的特征
        spatial_shapes = []  # 存储各尺度特征的空间形状
        for lvl, feat in enumerate(mlvl_feats):
            # 获取当前尺度特征的形状信息
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # 扁平化并调整维度，形状从[bs, num_cam, c, h, w]变为[num_cam, bs, h*w, c]
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # 根据配置添加相机嵌入
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            # 添加级别嵌入
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            # 保存空间形状和扁平化特征
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        # 在空间维度上拼接所有尺度的特征
        feat_flatten = torch.cat(feat_flatten, 2)  # 形状为[num_cam, bs, sum(h*w), c]
        # 转换空间形状为张量
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        # 计算每个尺度特征在拼接后的起始索引
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # 调整特征维度，准备输入编码器，形状变为[num_cam, sum(h*w), bs, embed_dims]
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        # 调用编码器生成BEV特征
        bev_embed = self.encoder(
            bev_queries,  # 输入查询 [bev_h*bev_w, bs, embed_dims]
            feat_flatten,  # 键特征 [num_cam, sum(h*w), bs, embed_dims]
            feat_flatten,  # 值特征 [num_cam, sum(h*w), bs, embed_dims]
            bev_h=bev_h,  # BEV高度
            bev_w=bev_w,  # BEV宽度
            bev_pos=bev_pos,  # BEV位置编码 [bev_h*bev_w, bs, embed_dims]
            spatial_shapes=spatial_shapes,  # 各尺度特征空间形状
            level_start_index=level_start_index,  # 各尺度特征起始索引
            prev_bev=prev_bev,  # 前一帧BEV特征
            shift=shift,  # BEV网格位移 [bs, 2]
            img_metas=img_metas,  # 图像元数据
        )

        return bev_embed

    def get_states_and_refs(
        self,
        bev_embed,
        object_query_embed,
        bev_h,
        bev_w,
        reference_points,
        reg_branches=None,
        cls_branches=None,
        img_metas=None
    ):
        """
        从BEV特征中获取状态和参考点信息，用于目标检测任务
        
        Args:
            bev_embed (Tensor): BEV（鸟瞰图）特征，形状为 [bev_h*bev_w, bs, embed_dims]
            object_query_embed (Tensor): 对象查询嵌入，形状为 [num_queries, 2*embed_dims]
            bev_h (int): BEV特征图的高度
            bev_w (int): BEV特征图的宽度
            reference_points (Tensor): 参考点，形状为 [num_queries, 3]
            reg_branches (nn.ModuleList, optional): 回归分支，用于预测目标位置，默认为None
            cls_branches (nn.ModuleList, optional): 分类分支，用于预测目标类别，默认为None
            img_metas (list[dict], optional): 图像元数据列表，包含图像相关信息，默认为None
        
        Returns:
            tuple[Tensor, Tensor, Tensor]: 
                - inter_states: 解码器中间状态，形状为 [num_dec_layers, num_queries, bs, embed_dims]
                - init_reference_out: 初始参考点，形状为 [bs, num_queries, 3]
                - inter_references_out: 解码器中间参考点，形状为 [num_dec_layers, bs, num_queries, 3]
        """
        
        # 获取批次大小
        bs = bev_embed.shape[1]
        
        # 将对象查询嵌入分割为查询位置和查询向量，各占一半维度
        # query_pos: 查询位置编码，形状为 [num_queries, embed_dims]
        # query: 查询向量，形状为 [num_queries, embed_dims]
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        
        # 扩展查询位置和查询向量到批次维度
        # query_pos: 形状变为 [bs, num_queries, embed_dims]
        # query: 形状变为 [bs, num_queries, embed_dims]
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        # 扩展参考点到批次维度并应用sigmoid函数归一化到[0,1]范围
        # reference_points: 形状从 [num_queries, 3] 变为 [bs, num_queries, 3]
        reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.sigmoid()

        # 保存初始参考点用于输出
        init_reference_out = reference_points
        
        # 调整查询和查询位置的维度以适应解码器输入要求
        # query: 形状从 [bs, num_queries, embed_dims] 变为 [num_queries, bs, embed_dims]
        # query_pos: 形状从 [bs, num_queries, embed_dims] 变为 [num_queries, bs, embed_dims]
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        
        # 调用解码器进行解码，生成中间状态和中间参考点
        # inter_states: 解码器各层输出状态，形状为 [num_dec_layers, num_queries, bs, embed_dims]
        # inter_references: 解码器各层参考点，形状为 [num_dec_layers, bs, num_queries, 3]
        inter_states, inter_references = self.decoder(
            query=query,                      # 查询向量
            key=None,                         # 键（BEV解码中不需要）
            value=bev_embed,                  # 值（BEV特征）
            query_pos=query_pos,              # 查询位置编码
            reference_points=reference_points,# 参考点
            reg_branches=reg_branches,        # 回归分支
            cls_branches=cls_branches,        # 分类分支
            # BEV特征的空间形状，仅包含一个级别 [bev_h, bev_w]
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            # BEV特征的级别起始索引，仅包含一个级别，起始索引为0
            level_start_index=torch.tensor([0], device=query.device),
            img_metas=img_metas# 图像元数据
        )
        
        # 保存中间参考点用于输出
        inter_references_out = inter_references

        # 返回解码器中间状态、初始参考点和中间参考点
        return inter_states, init_reference_out, inter_references_out

    # NOTE: we add a forward function to adaptive bevformer
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                real_h,
                real_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            real_h,
            real_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out