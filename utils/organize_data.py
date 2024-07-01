import os
import shutil
import random

def split_market1501(download_path, train_ratio=0.8):
    save_path = os.path.join(download_path, 'pytorch')
    train_path = os.path.join(download_path, 'bounding_box_train')
    train_save_path = os.path.join(save_path, 'train')
    val_save_path = os.path.join(save_path, 'val')

    # Create directories if they don't exist
    for path in [save_path, train_save_path, val_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Get all person IDs
    person_ids = set()
    for name in os.listdir(train_path):
        if name.endswith('.jpg'):
            person_id = name.split('_')[0]
            person_ids.add(person_id)

    # Randomly split person IDs into train and val sets
    person_ids = list(person_ids)
    random.shuffle(person_ids)
    split_idx = int(len(person_ids) * train_ratio)
    train_ids = set(person_ids[:split_idx])
    val_ids = set(person_ids[split_idx:])

    # Copy files to respective folders
    for name in os.listdir(train_path):
        if name.endswith('.jpg'):
            person_id = name.split('_')[0]
            src_path = os.path.join(train_path, name)
            
            if person_id in train_ids:
                dst_dir = os.path.join(train_save_path, person_id)
            elif person_id in val_ids:
                dst_dir = os.path.join(val_save_path, person_id)
            else:
                continue  # Skip if ID is not in either set (shouldn't happen)
            
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            
            dst_path = os.path.join(dst_dir, name)
            shutil.copy(src_path, dst_path)

    print(f"Split complete. Train set: {len(train_ids)} IDs, Validation set: {len(val_ids)} IDs")
    print(f"Train images: {sum(len(files) for _, _, files in os.walk(train_save_path))}")
    print(f"Validation images: {sum(len(files) for _, _, files in os.walk(val_save_path))}")

# Usage
download_path = './datasets/market1501/Market-1501-v15.09.15'
split_market1501(download_path)