from torch.utils.data import Subset
import numpy as np
from collections import defaultdict

def split_dataset_by_id(dataset, train_ratio=0.8):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        label_to_indices[label].append(idx)
    
    unique_labels = list(label_to_indices.keys())
    np.random.shuffle(unique_labels)
    
    train_size = int(len(unique_labels) * train_ratio)
    train_labels = unique_labels[:train_size]
    val_labels = unique_labels[train_size:]
    
    train_indices = [idx for label in train_labels for idx in label_to_indices[label]]
    val_indices = [idx for label in val_labels for idx in label_to_indices[label]]
    
    # Error checks and debugging information
    print(f"Total number of unique labels: {len(unique_labels)}")
    print(f"Number of training labels: {len(train_labels)}")
    print(f"Number of validation labels: {len(val_labels)}")
    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of validation samples: {len(val_indices)}")
    
    # Check for label overlap
    label_overlap = set(train_labels) & set(val_labels)
    if label_overlap:
        raise ValueError(f"Found {len(label_overlap)} overlapping labels between train and val sets")
    
    # Check for index overlap
    index_overlap = set(train_indices) & set(val_indices)
    if index_overlap:
        raise ValueError(f"Found {len(index_overlap)} overlapping indices between train and val sets")
    
    # Check if all indices are accounted for
    all_indices = set(range(len(dataset)))
    used_indices = set(train_indices + val_indices)
    if all_indices != used_indices:
        missing_indices = all_indices - used_indices
        extra_indices = used_indices - all_indices
        print(f"Warning: {len(missing_indices)} indices are missing from the split")
        print(f"Warning: {len(extra_indices)} extra indices are in the split")
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset