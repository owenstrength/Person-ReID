import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils.market1501 import Market1501Dataset
from models.oreid import OReID

# Data preprocessing
transform_test = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = Market1501Dataset(root_dir='./datasets/market1501/Market-1501-v15.09.15/pytorch/val', transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the best model
model = OReID(num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load('best_reid_model.pth'))
model.to(device)
model.eval()

def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            feature, _ = model(data)
            features.append(feature.cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(features), np.concatenate(labels)

def compute_distance_matrix(query_features, gallery_features):
    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
    dist_matrix = 2 - 2 * np.dot(query_features, gallery_features.T)
    return dist_matrix

def compute_map_and_cmc(dist_matrix, query_labels, gallery_labels, top_k):
    num_queries, num_gallery = dist_matrix.shape
    indices = np.argsort(dist_matrix, axis=1)
    
    matches = (gallery_labels[indices] == query_labels[:, np.newaxis]).astype(np.int32)
    
    # Compute CMC
    cmc = matches.cumsum(axis=1).mean(axis=0)
    cmc = cmc[:top_k]
    
    # Compute mAP
    num_relevant = (query_labels[:, np.newaxis] == gallery_labels).sum(axis=1)
    tmp_cmc = matches.cumsum(axis=1)
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = np.stack(tmp_cmc, axis=1)
    ap = tmp_cmc.sum(axis=1) / num_relevant
    map_score = ap.mean()
    
    return map_score, cmc

# Extract features for the test set
test_features, test_labels = extract_features(model, test_loader)

# Compute distance matrix
dist_matrix = compute_distance_matrix(test_features, test_features)

# Compute mAP and CMC
top_k = 50
map_score, cmc = compute_map_and_cmc(dist_matrix, test_labels, test_labels, top_k)

print(f'Test mAP: {map_score:.4f}')
print(f'Test CMC@1: {cmc[0]:.4f}')
print(f'Test CMC@5: {cmc[4]:.4f}')
print(f'Test CMC@10: {cmc[9]:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(range(1, top_k + 1), cmc)
plt.xlabel('Rank')
plt.ylabel('Matching Rate')
plt.title('Cumulative Matching Characteristic (CMC) Curve')
plt.grid(True)
plt.savefig('cmc_curve.png')
plt.close()  # Close the figure to free up memory

# Visualize some test results
def visualize_rankings(model, test_loader, num_queries=5, top_k=5):
    model.eval()
    all_features = []
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            features, _ = model(images)
            all_features.append(features.cpu())
            all_images.append(images.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute distance matrix
    dist_matrix = compute_distance_matrix(all_features.numpy(), all_features.numpy())
    
    plt.figure(figsize=(15, 3 * num_queries))
    
    for i in range(num_queries):
        query_idx = np.random.randint(0, len(all_labels))
        query_image = all_images[query_idx]
        query_label = all_labels[query_idx]
        
        distances = dist_matrix[query_idx]
        sorted_indices = np.argsort(distances)
        
        plt.subplot(num_queries, top_k + 1, i * (top_k + 1) + 1)
        plt.imshow(query_image.permute(1, 2, 0))
        plt.title(f'Query (ID: {query_label})')
        plt.axis('off')
        
        for j in range(top_k):
            matched_idx = sorted_indices[j + 1]  # +1 to skip the query image itself
            matched_image = all_images[matched_idx]
            matched_label = all_labels[matched_idx]
            
            plt.subplot(num_queries, top_k + 1, i * (top_k + 1) + j + 2)
            plt.imshow(matched_image.permute(1, 2, 0))
            color = 'green' if matched_label == query_label else 'red'
            plt.title(f'Rank {j+1} (ID: {matched_label})', color=color)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('ranking_visualization.png')
    plt.close()

# Visualize some test results
visualize_rankings(model, test_loader, num_queries=5, top_k=5)