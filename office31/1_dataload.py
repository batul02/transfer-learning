import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

# Paths
root_path = "./office31/"

# Classes to include
selected_classes = ['back_pack', 'bike', 'calculator', 'keyboard', 'laptop_computer', 'monitor']

# Define a simple transform (later you can do augmentations if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_filtered_office31(domain, selected_classes, batch_size=64, root_path=root_path):
    dataset_path = os.path.join(root_path, domain)

    # Load full dataset
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

    # Map selected class names to their class indices
    selected_indices = [full_dataset.class_to_idx[cls] for cls in selected_classes]

    # Filter dataset to include only selected classes
    filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in selected_indices]

    filtered_dataset = Subset(full_dataset, filtered_indices)

    loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return loader, selected_classes

# Load source and target
amazon_loader, _ = load_filtered_office31("amazon", selected_classes, batch_size=64)
dslr_loader, _ = load_filtered_office31("dslr", selected_classes, batch_size=64)

print(f"Amazon batches: {len(amazon_loader)}, DSLR batches: {len(dslr_loader)}")

# Collect source (Amazon)
X_amazon, y_amazon = [], []
for imgs, labels in amazon_loader:
    X_amazon.append(imgs)
    y_amazon.append(labels)
X_amazon = torch.cat(X_amazon)
y_amazon = torch.cat(y_amazon)

# Collect target (DSLR)
X_dslr, y_dslr = [], []
for imgs, labels in dslr_loader:
    X_dslr.append(imgs)
    y_dslr.append(labels)
X_dslr = torch.cat(X_dslr)
y_dslr = torch.cat(y_dslr)

print("Source (Amazon) shape:", X_amazon.shape, y_amazon.shape)
print("Target (DSLR) shape:", X_dslr.shape, y_dslr.shape)

# Save as a single file
torch.save({
    "X_amazon": X_amazon,
    "y_amazon": y_amazon,
    "X_dslr": X_dslr,
    "y_dslr": y_dslr,
}, "office31_filtered_amazon_dslr.pt")
