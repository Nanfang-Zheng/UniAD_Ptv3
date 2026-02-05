import numpy as np
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class GridSample_migrate(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=True,  # Default to True for PTv3
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, results):
        # Adapt for mmdet3d pipeline: Extract coords from 'points'
        if 'points' in results:
            # Assume results['points'] is a BasePoints object or similar wrapper
            # Convert to numpy for processing
            points = results['points']
            if hasattr(points, 'tensor'):
                coord = points.tensor[:, :3].numpy()
            else:
                coord = points[:, :3] # Assume it's already numpy or similar
        elif 'coord' in results:
             coord = results['coord']
        else:
            raise KeyError("GridSample_migrate requires 'points' or 'coord' in results")

        scaled_coord = coord / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            
            # Handle sampled_index if present (from Pointcept logic, kept for compatibility)
            if "sampled_index" in results:
                idx_unique = np.unique(
                    np.append(idx_unique, results["sampled_index"])
                )
                # Note: mmdet3d usually doesn't have 'segment' in results at this stage like Pointcept
                # Skipping mask logic unless 'segment' exists
                if "segment" in results:
                    mask = np.zeros_like(results["segment"]).astype(bool)
                    mask[results["sampled_index"]] = True
                    results["sampled_index"] = np.where(mask[idx_unique])[0]
            
            # Apply selection to points
            if 'points' in results:
                results['points'] = results['points'][idx_unique]
            
            # Also filter other keys if necessary? 
            # In mmdet3d, we usually filter points, and other fields like gt_bboxes are not per-point.
            # But if there are per-point labels (pts_semantic_mask), they should be filtered.
            if 'pts_semantic_mask' in results and results['pts_semantic_mask'] is not None:
                results['pts_semantic_mask'] = results['pts_semantic_mask'][idx_unique]

            if self.return_inverse:
                results["inverse"] = np.zeros_like(inverse)
                results["inverse"][idx_sort] = inverse
            
            if self.return_grid_coord:
                results["grid_coord"] = grid_coord[idx_unique]
            
            if self.return_min_coord:
                results["min_coord"] = min_coord.reshape([1, 3])
            
            if self.return_displacement:
                # Re-extract scaled_coord for selected points if necessary, 
                # but scaled_coord above corresponds to original points.
                # We need displacement for unique points.
                displacement = (
                    scaled_coord[idx_unique] - grid_coord[idx_unique] - 0.5
                )
                if self.project_displacement and "normal" in results:
                     # Assume normal is available and filtered
                     # This part is tricky if normal isn't filtered yet.
                     # Skipping complex projection logic for minimal migration unless needed.
                     pass
                results["displacement"] = displacement
            
            return results

        elif self.mode == "test":  # test mode
            # In mmdet3d test mode, we might not want to split into parts like Pointcept does
            # unless we are doing TTA or chunked inference.
            # For minimal migration compatible with UniAD, we likely just want voxelization 
            # without random sampling, OR we keep the whole scene.
            # PTv3 test mode in Pointcept returns a LIST of data parts.
            # This might break mmdet3d pipeline which expects a single dict.
            # Strategy: For now, implement 'train' logic (random sample per voxel) even for test
            # OR just take the first point per voxel (deterministic).
            # Let's use deterministic selection for test (first point).
            
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]
            
            if 'points' in results:
                results['points'] = results['points'][idx_unique]
            if 'pts_semantic_mask' in results and results['pts_semantic_mask'] is not None:
                results['pts_semantic_mask'] = results['pts_semantic_mask'][idx_unique]

            if self.return_grid_coord:
                results["grid_coord"] = grid_coord[idx_unique]
            
            return results
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr
