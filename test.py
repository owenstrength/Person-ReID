import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.market1501 import Market1501Dataset
from models.resnetreid import ResNetReID
from utils.misc import compute_distance_matrix, compute_map_and_cmc

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32

# Data transforms
transform_val = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # This is the mean and std deviation of Market-1501 dataset
])

# Load test dataset
test_dataset = Market1501Dataset(root_dir='./datasets/market1501/Market-1501-v15.09.15/pytorch/val', transform=transform_val)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Load the trained model
model = ResNetReID(num_classes=600).to(device)
model.load_state_dict(torch.load('best_reid_model.pth'))
model.eval()

def extract_features(model, dataloader):
    """
    Extracts features from a given model using the provided dataloader.

    Args:
        model (torch.nn.Module): The model to extract features from.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the data.

    Returns:
        tuple: A tuple containing the concatenated features and labels.
            - features (numpy.ndarray): The concatenated features.
            - labels (numpy.ndarray): The concatenated labels.
    """
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images, batch_labels = batch
            images = images.to(device)
            batch_features = model(images)
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels)

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize an image tensor to convert it back to the original pixel value range [0, 1].
    
    Args:
    tensor (torch.Tensor): Normalized image tensor of shape (C, H, W) or (B, C, H, W)
    mean (list): Mean values used for normalization
    std (list): Standard deviation values used for normalization
    
    Returns:
    torch.Tensor: Denormalized image tensor
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    
    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized.squeeze(0) if denormalized.size(0) == 1 else denormalized

def plot_cmc_curve(cmc):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cmc) + 1), cmc)
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.title('Cumulative Matching Characteristic (CMC) Curve')
    plt.grid(True)
    plt.savefig('cmc_curve.png')
    plt.close()

def visualize_rankings(model, dataloader, num_queries=5, top_k=5):
    model.eval()
    all_features, all_labels = extract_features(model, dataloader)
    dist_matrix = compute_distance_matrix(all_features)
    
    plt.figure(figsize=(15, 3 * num_queries))
    
    for i in range(num_queries):
        query_idx = np.random.randint(0, len(all_labels))
        query_label = all_labels[query_idx]
        
        distances = dist_matrix[query_idx]
        sorted_indices = np.argsort(distances)
        
        for j in range(top_k + 1):
            if j == 0:
                plt.subplot(num_queries, top_k + 1, i * (top_k + 1) + 1)
                plt.title(f'Query (ID: {query_label})')
            else:
                plt.subplot(num_queries, top_k + 1, i * (top_k + 1) + j + 1)
                matched_idx = sorted_indices[j]
                matched_label = all_labels[matched_idx]
                color = 'green' if matched_label == query_label else 'red'
                plt.title(f'Rank {j} (ID: {matched_label})', color=color)
            
            img_idx = sorted_indices[j] if j > 0 else query_idx
            img, _ = dataloader.dataset[img_idx]
            img = denormalize_image(img)
            plt.imshow(img.permute(1, 2, 0))
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('ranking_visualization.png')
    plt.close()

# Extract features
test_features, test_labels = extract_features(model, test_loader)

# Compute distance matrix
dist_matrix = compute_distance_matrix(test_features)

# Compute mAP and CMC
top_k = 50
map_score, cmc = compute_map_and_cmc(dist_matrix, test_labels, test_labels, top_k)

print(f'Test mAP: {map_score:.4f}')
print(f'Test CMC@1: {cmc[0]:.4f}')
print(f'Test CMC@5: {cmc[4]:.4f}')
print(f'Test CMC@10: {cmc[9]:.4f}')

# Plot CMC curve
plot_cmc_curve(cmc)

# Visualize some test results
visualize_rankings(model, test_loader, num_queries=5, top_k=5)

print("Testing completed. CMC curve and ranking visualizations have been saved.")