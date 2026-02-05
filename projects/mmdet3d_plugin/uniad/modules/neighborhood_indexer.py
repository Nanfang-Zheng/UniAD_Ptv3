import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from projects.mmdet3d_plugin.models.backbones.ptv3_models.serialization import encode

class BaseNeighborhoodIndexer(ABC):
    @abstractmethod
    def build(self, lidar_point_dict, min_coord=None):
        """
        Build the index structure from the LiDAR point cloud.
        Args:
            lidar_point_dict (dict or Point): Output from PTv3 backbone.
            min_coord (torch.Tensor, optional): (B, 3) or (1, 3) min coords used for grid sampling.
        """
        pass

    @abstractmethod
    def search(self, query_coords, query_batch_idx, window_size=None):
        """
        Search for neighbors.
        Args:
            query_coords: (M, 3) Metric coordinates
            query_batch_idx: (M,) Batch indices
            window_size: int, optional override
        Returns:
            neighbor_feats: (M, K, C) or packed
            neighbor_mask: (M, K)
        """
        pass

class PTv3SerializationWindowIndexer(BaseNeighborhoodIndexer):
    def __init__(self, order_index=0, window_size=256, depth=16, order="z", grid_size=0.05):
        self.order_index = order_index
        self.window_size = window_size
        self.depth = depth
        self.order = order
        self.grid_size = grid_size
        
        self.codes = None
        self.feat_sorted = None
        self.coord_sorted = None
        self.min_coord = None

    def build(self, lidar_point_dict, min_coord=None):
        # Extract metadata
        self.codes = lidar_point_dict['serialized_code'][self.order_index] # (N,) unsorted codes
        sort_idx = lidar_point_dict['serialized_order'][self.order_index] # (N,) sorting indices
        
        # Sort data
        self.feat_sorted = lidar_point_dict['feat'][sort_idx]
        self.coord_sorted = lidar_point_dict['coord'][sort_idx]
        self.codes = self.codes[sort_idx] # Now sorted
        
        self.min_coord = min_coord

    def search(self, query_coords, query_batch_idx, window_size=None):
        if window_size is None:
            window_size = self.window_size
        
        device = query_coords.device
        num_queries = query_coords.shape[0]
        
        # 1. Quantize Query Coords
        # query_coords: (M, 3)
        # We need to subtract min_coord.
        # min_coord is (B, 3) or (1, 3). 
        # If min_coord is per batch, we need to gather.
        
        q_grid = query_coords.clone()
        if self.min_coord is not None:
            if self.min_coord.shape[0] > 1:
                # Per batch min_coord
                batch_mins = self.min_coord[query_batch_idx]
                q_grid -= batch_mins
            else:
                q_grid -= self.min_coord
                
        q_grid = torch.floor(q_grid / self.grid_size).int()
        
        # 2. Encode
        query_codes = encode(q_grid, query_batch_idx, depth=self.depth, order=self.order)
        
        # 3. Binary Search
        # Find insertion positions
        pos = torch.searchsorted(self.codes, query_codes)
        
        # 4. Window Selection
        # Create indices [pos-W, pos+W]
        # We need to handle boundary conditions (0 and N)
        # Simple approach: Create a window offset tensor
        K = 2 * window_size + 1
        offsets = torch.arange(-window_size, window_size + 1, device=device)
        
        # (M, K)
        neighbor_indices = pos.unsqueeze(1) + offsets.unsqueeze(0)
        
        # Clamp indices
        max_idx = self.codes.shape[0] - 1
        neighbor_indices = neighbor_indices.clamp(0, max_idx)
        
        # Gather features
        # (M, K, C)
        neighbor_feats = self.feat_sorted[neighbor_indices]
        
        # Create Mask
        # If the index was clamped, it might duplicate features.
        # But conceptually, we just want to attend to "available" neighbors.
        # Since we clamped, we are attending to valid points, but they might be far away 
        # (if we clamped from far outside).
        # Better mask: Check if code difference is reasonable? 
        # Or just use all clamped points. 
        # For strict window attention, usually we don't mask unless we cross batch boundaries.
        # But sorted codes group batches together.
        # So [pos-W, pos+W] might cross batch boundary!
        
        # Batch Check
        # (M, K)
        # neighbor_batch = self.batch_sorted[neighbor_indices] # We didn't store batch_sorted
        # But codes contain batch info in high bits!
        # code = batch << depth*3 | spatial_code
        # So we can check if neighbor_code >> (depth*3) == query_batch_idx
        
        neighbor_codes = self.codes[neighbor_indices]
        neighbor_batches = neighbor_codes >> (self.depth * 3)
        
        # Mask out points from different batches
        mask = (neighbor_batches == query_batch_idx.unsqueeze(1))
        
        return neighbor_feats, mask

class GridHashBucketIndexer(BaseNeighborhoodIndexer):
    def __init__(self, grid_size=0.05):
        self.grid_size = grid_size
        self.hash_table = {} # simplistic python dict for prototype
        self.feats = None
        self.min_coord = None

    def build(self, lidar_point_dict, min_coord=None):
        self.feats = lidar_point_dict['feat']
        self.min_coord = min_coord
        coords = lidar_point_dict['grid_coord'] # Already quantized
        batch = lidar_point_dict['batch']
        
        # Build hash map
        # Using a simple coordinate hash
        # key = (b, x, y, z)
        # This is slow in pure python for dense points.
        # For prototype, we assume small scale or use PTv3's hash logic if possible.
        # Let's use a very simple spatial hash for prototype correctness.
        pass

    def search(self, query_coords, query_batch_idx, window_size=None):
        # Prototype placeholder
        return torch.zeros((query_coords.shape[0], 1, self.feats.shape[1]), device=query_coords.device), \
               torch.zeros((query_coords.shape[0], 1), dtype=torch.bool, device=query_coords.device)

