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

def compute_distance_matrix(features):
    """
    Compute the distance matrix between feature vectors.

    Args:
        features (numpy.ndarray): Array of feature vectors.

    Returns:
        numpy.ndarray: Distance matrix.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_normalized = features / norms

    dot_product = np.dot(features_normalized, features_normalized.T)
    dist_matrix = np.sqrt(2 - 2 * dot_product)

    dist_matrix = np.maximum(dist_matrix, 0)

    np.fill_diagonal(dist_matrix, np.inf)
    
    return dist_matrix

def compute_map_and_cmc(dist_matrix, query_labels, gallery_labels, top_k):
    """
    Compute Mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) given a distance matrix,
    query labels, gallery labels, and top-k value.

    Args:
        dist_matrix (numpy.ndarray): The distance matrix of shape (num_queries, num_gallery) containing pairwise distances between queries and gallery samples.
        query_labels (numpy.ndarray): The labels of the query samples of shape (num_queries,).
        gallery_labels (numpy.ndarray): The labels of the gallery samples of shape (num_gallery,).
        top_k (int): The maximum rank to consider for computing CMC.

    Returns:
        tuple: A tuple containing the mAP score and the CMC values.

    """
    num_queries, num_gallery = dist_matrix.shape
    indices = np.argsort(dist_matrix, axis=1)
    matches = (gallery_labels[indices] == query_labels[:, np.newaxis]).astype(np.int32)

    # Compute CMC
    cmc = np.zeros(top_k)
    for i in range(num_queries):
        rank = np.where(matches[i] == 1)[0]
        if rank.size > 0:
            rank = rank[0]
            if rank < top_k:
                cmc[rank:] += 1
    cmc = cmc / num_queries

    # Compute mAP
    ap = np.zeros(num_queries)
    for i in range(num_queries):
        relevant = matches[i]
        if relevant.sum() > 0:
            cumsum = np.cumsum(relevant)
            precision = cumsum / (np.arange(num_gallery) + 1)
            ap[i] = (precision * relevant).sum() / relevant.sum()
    
    map_score = ap.mean()

    return map_score, cmc