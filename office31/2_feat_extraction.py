import torchvision.models as models
import torch.nn as nn
import torch

data = torch.load("office31_filtered_amazon_dslr.pt")
X_amazon = data["X_amazon"]
y_amazon = data["y_amazon"]
X_dslr = data["X_dslr"]
y_dslr = data["y_dslr"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50.fc = nn.Identity()  # Remove final classification layer
resnet50 = resnet50.to(device)
resnet50.eval()

def extract_features(model, data):
    features = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(data, batch_size=64):
            batch = batch.to(device)
            feats = model(batch)
            features.append(feats.cpu())
    return torch.cat(features, dim=0)

X_amazon_feats = extract_features(resnet50, X_amazon)
X_dslr_feats = extract_features(resnet50, X_dslr)

print("Shape of Amazon feat: ", X_amazon_feats.shape)
print("Shape of dslr feat: ", X_dslr_feats.shape)

torch.save({
    "X_amazon_feats": X_amazon_feats,
    "y_amazon": y_amazon,
    "X_dslr_feats": X_dslr_feats,
    "y_dslr": y_dslr
}, 'extracted_features.pt')

print("Feature extraction complete and saved.")

