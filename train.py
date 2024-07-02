import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

from utils.market1501 import Market1501Dataset
from utils.misc import compute_distance_matrix, compute_map_and_cmc, split_dataset_by_id
from models.resnetreid import ResNetReID, CrossEntropyLabelSmooth


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4

# Data transforms
transform_train = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # This is the mean and std deviation of Market-1501 dataset
])

transform_val = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # This is the mean and std deviation of Market-1501 dataset
])

# Load and split dataset
full_dataset = Market1501Dataset(root_dir='./datasets/market1501/Market-1501-v15.09.15/pytorch/train', transform=transform_train)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = split_dataset_by_id(full_dataset, train_ratio=0.8)
val_dataset.dataset.transform = transform_val

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Initialize model, loss, and optimizer
model = ResNetReID(num_classes=len(full_dataset.classes)).to(device)
criterion = CrossEntropyLabelSmooth(num_classes=len(full_dataset.classes))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def train(model, dataloader, criterion, optimizer, device, epoch=0):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False,  bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        features, outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, epoch=0):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating Epoch {epoch}", leave=False, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            features = model(images)
            outputs = model.classifier(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return epoch_loss, epoch_acc, all_features, all_labels

if os.path.exists('best_reid_model.pth'):
    model.load_state_dict(torch.load('best_reid_model.pth'))
    print("Loaded pre-trained model")


# Training loop
best_map = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch=epoch)
    val_loss, val_acc, val_features, val_labels = validate(model, val_loader, criterion, device, epoch=epoch)
    
    dist_matrix = compute_distance_matrix(val_features)
    map_score, cmc = compute_map_and_cmc(dist_matrix, val_labels, val_labels, top_k=10)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"mAP: {map_score:.4f}, Rank-1: {cmc[0]:.4f}, Rank-5: {cmc[4]:.4f}, Rank-10: {cmc[9]:.4f}")
    
    if map_score > best_map:
        best_map = map_score
        torch.save(model.state_dict(), 'best_reid_model.pth')
        print("Saved new best model")
    
    print()

print(f"Training completed. Best mAP: {best_map:.4f}")