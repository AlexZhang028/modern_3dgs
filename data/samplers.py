import torch
from typing import Optional, Any, Tuple
from torch.utils.data import DataLoader

def identity_collate_fn(batch):
    """
    Collate function to unpack the batch list when batch_size=1.
    Replaces lambda x: x[0] which is not picklable on Windows.
    """
    return batch[0]

class DataSampler:
    """
    Base class for Data Sampler (Strategy Pattern).
    isolates sampling differences between static and temporal modes.
    """
    
    def sample(self) -> Tuple[Any, Optional[float]]:
        """
        Sample a training example.
        
        Returns:
            (camera, timestamp): Camera and timestamp (static mode returns None).
        """
        raise NotImplementedError


class StaticSampler(DataSampler):
    """Sampler for static scenes using DataLoader for prefetching."""
    
    def __init__(self, dataset, num_workers: int = 0):
        self.dataset = dataset
        self.num_cameras = len(dataset)
        
        # Use DataLoader for parallel IO if workers > 0, otherwise standard behavior
        # Note: Even with 0 workers, DataLoader provides a unified interface, but we just use it for >0
        self.use_dataloader = num_workers > 0
        
        if self.use_dataloader:
            print(f"StaticSampler: Enabled DataLoader with {num_workers} workers.")
            self.loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=identity_collate_fn, # Return single item directly
                persistent_workers=True
            )
            self.loader_iter = iter(self.loader)
        else:
            print("StaticSampler: Main thread data loading (0 workers).")
            self.viewpoint_stack = []
    
    def sample(self) -> Tuple[Any, None]:
        """Epoch-based Random Sampling."""
        if self.use_dataloader:
            try:
                sample = next(self.loader_iter)
            except StopIteration:
                self.loader_iter = iter(self.loader)
                sample = next(self.loader_iter)
            
            return sample["camera"], None
        else:
            # Refill stack if empty
            if not self.viewpoint_stack:
                self.viewpoint_stack = torch.randperm(self.num_cameras).tolist()
            
            idx = self.viewpoint_stack.pop()
            sample = self.dataset[idx]
            camera = sample["camera"]
            return camera, None


class TemporalSampler(DataSampler):
    """Sampler for temporal scenes (FreeTimeGS)."""
    
    def __init__(self, dataset, num_workers: int = 0):
        self.dataset = dataset
        self.num_cameras = len(dataset)
        self.use_dataloader = num_workers > 0
        
        if self.use_dataloader:
            print(f"TemporalSampler: Enabled DataLoader with {num_workers} workers.")
            self.loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=True, # Random sampling
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=identity_collate_fn,
                persistent_workers=True
            )
            self.loader_iter = iter(self.loader)
        else:
            print("TemporalSampler: Main thread data loading (0 workers).")
            self.viewpoint_stack = []
    
    def sample(self) -> Tuple[Any, float]:
        """Randomly sample camera and timestamp."""
        if self.use_dataloader:
            try:
                sample = next(self.loader_iter)
            except StopIteration:
                self.loader_iter = iter(self.loader)
                sample = next(self.loader_iter)
            
            camera = sample["camera"]
        else:
            # Refill stack if empty
            if not self.viewpoint_stack:
                self.viewpoint_stack = torch.randperm(self.num_cameras).tolist()
            
            idx = self.viewpoint_stack.pop()
            sample = self.dataset[idx]
            camera = sample["camera"]

        # Timestamp Handling: Use camera's timestamp if available (reconstruction), else random (generative?)
        # For FreeTimeGS reconstruction, we MUST use the frame time.
        timestamp = camera.timestamp
        # assert 0.0 <= timestamp <= 1.0, f"Error: Timestamp {timestamp} is out of range [0, 1]!"
        if timestamp is None:
             # Fallback if dataset has no timestamps (e.g. static dataset), random might be appropriate 
             # or we assume static t=0.5
             timestamp = torch.rand(1).item()
        
        return camera, timestamp
