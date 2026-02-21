
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D

@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        results['gt_map_masks'] = DC(
            to_tensor(results['gt_map_masks']), stack=True)

        return results


@PIPELINES.register_module()
class Ptv3CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """PTv3-specific formatting bundle.

    Keep `points` behavior from `DefaultFormatBundle3D` and force PTv3 fields
    (`grid_coord`, optional `min_coord`) to be non-stacked DataContainers so
    collate_fn does not create an extra batch dimension like (B, N, 3).
    """

    def __call__(self, results):
        results = super(Ptv3CustomDefaultFormatBundle3D, self).__call__(results)

        if 'grid_coord' in results:
            results['grid_coord'] = DC(to_tensor(results['grid_coord']), stack=False)
        if 'min_coord' in results:
            results['min_coord'] = DC(to_tensor(results['min_coord']), stack=False)
        if 'gt_map_masks' in results:
            results['gt_map_masks'] = DC(to_tensor(results['gt_map_masks']), stack=True)

        return results
