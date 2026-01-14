import torch
import numpy as np
from typing import Iterator, List, Dict, Any


class CamBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler for camera data that groups samples by timestamp.
    Each batch contains all camera views for a single timestamp.
    """
    
    def __init__(self, dataset, seed: int = 42):
        self.dataset = dataset
        self.base_seed = seed
        self.current_epoch = 0
        
        # Pre-compute grouped indices for efficiency
        self.grouped_indices = self._build_grouped_indices()
        self.timestamps = list(self.grouped_indices.keys())
        self.num_batches = len(self.timestamps)
        
        # Validate data integrity
        self._validate_groups()
    
    def _build_grouped_indices(self) -> Dict[Any, List[int]]:
        """Build grouped indices dictionary efficiently."""
        grouped_indices = {}
        
        for idx in range(len(self.dataset)):
            time_id = self.dataset[idx]['time_id']
            if time_id not in grouped_indices:
                grouped_indices[time_id] = []
            grouped_indices[time_id].append(idx)
        
        return grouped_indices
    
    def _validate_groups(self) -> None:
        """Validate that each timestamp has exactly 6 cameras."""
        for time_id, indices in self.grouped_indices.items():
            if len(indices) != 6:
                raise ValueError(
                    f"Timestamp {time_id} has {len(indices)} cameras, expected 6"
                )
    
    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for shuffling."""
        if epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}")
        self.current_epoch = epoch
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over batches of camera indices.
        Each batch contains indices for all 6 cameras of one timestamp.
        """
        # Create epoch-specific seed for reproducible shuffling
        epoch_seed = self.base_seed + self.current_epoch
        
        # Use numpy for better performance
        np_rng = np.random.RandomState(epoch_seed)
        
        # Create shuffled copy of timestamps
        timestamps_shuffled = self.timestamps.copy()
        np_rng.shuffle(timestamps_shuffled)
        
        # Yield batches
        for time_id in timestamps_shuffled:
            yield self.grouped_indices[time_id]
    
    def __len__(self) -> int:
        """Return the number of batches (timestamps)."""
        return self.num_batches
    
    def get_epoch_info(self) -> Dict[str, Any]:
        """Get information about current epoch state."""
        return {
            'current_epoch': self.current_epoch,
            'num_batches': self.num_batches,
            'num_timestamps': len(self.timestamps),
            'total_samples': len(self.dataset),
            'epoch_seed': self.base_seed + self.current_epoch
        }