# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """生成用于空间交叉注意力(SCA)和时间自注意力(TSA)的参考点。
        
        该方法根据指定的维度(dim)生成不同类型的参考点：
        - 3D参考点用于空间交叉注意力，从BEV空间的每个pillar中均匀采样点
        - 2D参考点用于时间自注意力，均匀分布在BEV平面上

        Args:
            H (int): BEV特征图的高度。
            W (int): BEV特征图的宽度。
            Z (int, optional): Pillar的高度，即BEV空间的高度范围。默认值为8。
            num_points_in_pillar (int, optional): 每个pillar中均匀采样的点数量。默认值为4。
            dim (str, optional): 参考点的维度，可以是'3d'或'2d'。默认值为'3d'。
            bs (int, optional): 批次大小。默认值为1。
            device (torch.device, optional): 参考点张量所在的设备。默认值为'cuda'。
            dtype (torch.dtype, optional): 参考点张量的数据类型。默认值为torch.float。

        Returns:
            Tensor: 生成的参考点张量。
                - 当dim='3d'时，形状为(bs, num_points_in_pillar*H*W, 3)，
                  表示每个批次中所有pillar的采样点坐标(x, y, z)。
                - 当dim='2d'时，形状为(bs, H*W, 1, 2)，
                  表示每个批次中BEV平面上的参考点坐标(x, y)。
        """

        # 生成3D空间中的参考点，用于空间交叉注意力(SCA)
        if dim == '3d':
            # 在Z轴方向上均匀采样num_points_in_pillar个点
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            
            # 在X轴方向上均匀采样W个点（宽度方向）
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            
            # 在Y轴方向上均匀采样H个点（高度方向）
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            
            # 堆叠x、y、z坐标，形成3D参考点 (num_points_in_pillar, H, W, 3)
            ref_3d = torch.stack((xs, ys, zs), -1)
            
            # 调整维度顺序并展平，得到形状为 (num_points_in_pillar, 3, H*W) 的张量
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2)
            
            # 再次调整维度顺序，得到形状为 (num_points_in_pillar, H*W, 3) 的张量
            ref_3d = ref_3d.permute(0, 2, 1)
            
            # 扩展到批次大小，得到最终形状 (bs, num_points_in_pillar*H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # 生成2D BEV平面上的参考点，用于时间自注意力(TSA)
        elif dim == '2d':
            # 在BEV平面上生成均匀分布的网格点
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            
            # 将y坐标展平并归一化到[0, 1]范围，形状为(1, H*W)
            ref_y = ref_y.reshape(-1)[None] / H
            
            # 将x坐标展平并归一化到[0, 1]范围，形状为(1, H*W)
            ref_x = ref_x.reshape(-1)[None] / W
            
            # 堆叠x、y坐标，形成2D参考点 (1, H*W, 2)
            ref_2d = torch.stack((ref_x, ref_y), -1)
            
            # 扩展到批次大小并增加一个维度，得到最终形状 (bs, H*W, 1, 2)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        """将BEV空间中的3D参考点转换为相机图像平面上的2D坐标，并生成有效掩码。
        
        该方法是BEVFormerEncoder的核心功能之一，负责将BEV空间中的参考点通过
        相机外参矩阵映射到图像平面，并生成指示哪些点在相机视野内的掩码。

        Args:
            reference_points (Tensor): BEV空间中的参考点，形状为(B, D, num_query, 3)，
                B为批次大小，D为每个pillar的采样点数量，num_query为查询点数量。
            pc_range (list): 点云的3D范围，格式为[x_min, y_min, z_min, x_max, y_max, z_max]。
            img_metas (list): 图像元数据列表，每个元素包含图像的相关信息，如lidar2img转换矩阵。

        Returns:
            tuple: 包含以下元素的元组：
                - reference_points_cam (Tensor): 相机图像平面上的参考点，
                  形状为(num_cam, B, num_query, D, 2)，num_cam为相机数量。
                - bev_mask (Tensor): BEV掩码，指示哪些点在相机视野内，
                  形状为(num_cam, B, num_query, D)。
        """
        # 从图像元数据中获取lidar到图像的转换矩阵
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)，N为相机数量
        
        # 克隆参考点以避免修改原始数据
        reference_points = reference_points.clone()

        # 将参考点从归一化坐标转换为实际3D坐标
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]  # x坐标
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]  # y坐标
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]  # z坐标

        # 将3D点转换为齐次坐标 (x, y, z, 1)
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        # 调整参考点的维度顺序 (B, D, num_query, 4) -> (D, B, num_query, 4)
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        # 扩展参考点以适应多个相机 (D, B, 1, num_query, 4) -> (D, B, num_cam, num_query, 4, 1)
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        # 扩展lidar2img矩阵以适应多个采样点和查询点 (1, B, num_cam, 1, 4, 4) -> (D, B, num_cam, num_query, 4, 4)
        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        # 应用lidar2img转换矩阵将3D点转换为相机坐标
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        
        eps = 1e-5  # 避免除零错误的小值

        # 生成BEV掩码：z坐标大于eps的点视为有效
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        
        # 进行透视除法得到2D图像坐标 (x/z, y/z)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # 将图像坐标归一化到[0, 1]范围
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]  # 宽度方向归一化
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]  # 高度方向归一化

        # 更新BEV掩码：只保留在图像视野内的点 (x∈[0,1], y∈[0,1]且z>eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        # 处理可能的NaN值
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        # 调整输出的维度顺序以适应后续处理
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)  # (num_cam, B, num_query, D, 2)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)  # (num_cam, B, num_query, D)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                img_metas=None,
                **kwargs):
        """BEVFormer编码器的前向传播函数。
        
        该方法实现了BEVFormer编码器的核心逻辑，通过处理BEV查询和多相机特征，
        生成融合了空间和时间信息的BEV特征表示。

        Args:
            bev_query (Tensor): 输入的BEV查询张量，形状为`(num_query, bs, embed_dims)`，
                num_query表示查询点数量，bs表示批次大小，embed_dims表示嵌入维度。
            key (Tensor): 输入的多相机特征键，形状为`(num_cam, num_value, bs, embed_dims)`，
                num_cam表示相机数量，num_value表示特征点数量。
            value (Tensor): 输入的多相机特征值，与key具有相同的形状。
            bev_h (int, optional): BEV特征图的高度。
            bev_w (int, optional): BEV特征图的宽度。
            bev_pos (Tensor, optional): BEV位置编码，形状与bev_query相同。
            spatial_shapes (Tensor, optional): 特征图的空间形状信息。
            level_start_index (Tensor, optional): 不同层级特征的起始索引。
            valid_ratios (Tensor, optional): 特征图上有效区域的比例，形状为`(bs, num_levels, 2)`。
            prev_bev (Tensor, optional): 上一帧的BEV特征，用于时间注意力机制。
            shift (float, optional): BEV特征的偏移量，用于处理时间一致性。
            img_metas (list, optional): 图像元数据列表，包含相机内外参等信息。
            *args: 其他位置参数。
            **kwargs: 其他关键字参数。

        Returns:
            Tensor: 编码器的输出结果。当return_intermediate为False时，
                形状为`(num_query, bs, embed_dims)`；否则形状为
                `(num_layers, num_query, bs, embed_dims)`，包含所有中间层的输出。
        """
        output = bev_query  # 初始化输出为输入的BEV查询
        intermediate = []  # 存储中间层输出的列表

        # 获取3D参考点，用于空间交叉注意力(SCA)
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, 
            dim='3d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
        
        # 获取2D参考点，用于时间自注意力(TSA)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), 
            device=bev_query.device, dtype=bev_query.dtype)

        # 将3D参考点投影到相机图像平面，生成相机参考点和BEV掩码
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas)

        # 处理2D参考点的偏移，用于处理时间一致性
        # NOTE: We have fixed this bug
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # 调整BEV查询和位置编码的维度顺序：(num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        
        # 获取2D参考点的形状信息
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        
        # 处理历史BEV特征，用于时间注意力
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            # 将历史BEV和当前查询拼接，用于混合注意力
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            # 生成混合2D参考点，包含偏移和当前参考点
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            # 如果没有历史BEV，使用当前参考点重复两次
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        # 逐层处理编码器层
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,        # BEV位置编码
                ref_2d=hybird_ref_2d,   # 混合2D参考点
                ref_3d=ref_3d,          # 3D参考点
                bev_h=bev_h,            # BEV高度
                bev_w=bev_w,            # BEV宽度
                spatial_shapes=spatial_shapes,  # 空间形状
                level_start_index=level_start_index,  # 层级起始索引
                reference_points_cam=reference_points_cam,  # 相机参考点
                bev_mask=bev_mask,      # BEV掩码
                prev_bev=prev_bev,      # 历史BEV特征
                **kwargs)

            # 更新当前查询为当前层的输出
            bev_query = output
            
            # 如果需要返回中间结果，将当前输出添加到列表
            if self.return_intermediate:
                intermediate.append(output)

        # 根据配置返回结果
        if self.return_intermediate:
            # 返回所有中间层的输出，形状为(num_layers, num_query, bs, embed_dims)
            return torch.stack(intermediate)

        # 只返回最后一层的输出
        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default: None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default: 2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        """BEVFormerLayer类的构造函数，初始化Transformer层的各个组件。
        
        该方法继承自MyCustomBaseTransformerLayer，配置并初始化BEVFormer中的Transformer层，
        包括注意力机制、前馈网络、归一化层等组件，并验证操作顺序的有效性。

        Args:
            attn_cfgs (list[ConfigDict] | list[dict] | dict): 注意力机制的配置，
                可以是自注意力或交叉注意力的配置列表，顺序应与operation_order一致。
                如果是单个dict，将扩展为与operation_order中注意力数量相同的配置。
            feedforward_channels (int): 前馈网络(FFN)的隐藏层通道数。
            ffn_dropout (float, optional): 前馈网络中的dropout概率，默认0.0。
            operation_order (tuple[str], optional): Transformer层中操作的执行顺序，
                例如('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')。
            act_cfg (dict, optional): 前馈网络中激活函数的配置，默认使用ReLU。
            norm_cfg (dict, optional): 归一化层的配置，默认使用LayerNorm。
            ffn_num_fcs (int, optional): 前馈网络中全连接层的数量，默认2层。
            **kwargs: 其他传递给父类构造函数的关键字参数。

        Raises:
            AssertionError: 如果operation_order的长度不等于6，或者包含的操作类型不完整。
        """
        # 调用父类构造函数，初始化Transformer层的基本组件
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        
        # 禁用FP16混合精度训练
        self.fp16_enabled = False
        
        # 验证操作顺序的长度必须为6（通常是self_attn -> norm -> cross_attn -> norm -> ffn -> norm）
        assert len(operation_order) == 6
        
        # 验证操作顺序必须包含所有必要的操作类型
        assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query