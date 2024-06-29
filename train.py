import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score

from utils.market1501 import Market1501Dataset
from models.oreid import OReID  # Make sure to import your custom model

# Data loading and preprocessing
transform_train = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Market 1501 dataset
full_dataset = Market1501Dataset(root_dir='./datasets/market1501/Market-1501-v15.09.15/pytorch/train', transform=transform_train)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = transform_val

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Model initialization
model = OReID(num_classes=len(full_dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training and validation loop
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_features(model, dataloader):
    model.eval()
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

# Training and validation loop
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_val_map = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_predictions = []
    train_targets = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        features, output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(output, 1)
        train_predictions.extend(predicted.cpu().numpy())
        train_targets.extend(target.cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

    train_accuracy = accuracy_score(train_targets, train_predictions)
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_targets = []

    # Extract features for validation set
    val_features, val_labels = extract_features(model, val_loader)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(val_features, val_features)

    # Compute mAP and CMC
    map_score, cmc = compute_map_and_cmc(dist_matrix, val_labels, val_labels, top_k=10)

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            features, output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(target.cpu().numpy())

    val_accuracy = accuracy_score(val_targets, val_predictions)
    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print(f'Validation mAP: {map_score:.4f}')
    print(f'Validation CMC: {cmc}')

    # Save the best model based on mAP
    if map_score > best_val_map:
        best_val_map = map_score
        torch.save(model.state_dict(), 'best_reid_model.pth')

    scheduler.step()

print(f'Best validation mAP: {best_val_map:.4f}')

# Save the final model
torch.save(model.state_dict(), 'final_reid_model.pth')