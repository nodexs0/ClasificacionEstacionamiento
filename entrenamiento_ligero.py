import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Configuración
base_path = "/content/drive/MyDrive/Datasets/CNR-EXT-Patches-150x150"
labels_path = os.path.join(base_path, "LABELS")
images_path = os.path.join(base_path, "PATCHES")
batch_size = 16  # Reducido para optimizar el rendimiento
num_epochs = 1
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset personalizado
class CNRDataset(Dataset):
    def __init__(self, txt_file, images_path, transform=None):
        self.images_path = images_path
        self.transform = transform
        self.data = []
        
        with open(txt_file, "r") as file:
            for line in file:
                img_path, label = line.strip().split()
                self.data.append((img_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_relative_path, label = self.data[idx]
        img_path = os.path.join(self.images_path, img_relative_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Mantener la resolución original
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar datasets
train_dataset = CNRDataset(txt_file=os.path.join(labels_path, "train.txt"), images_path=images_path, transform=transform)
val_dataset = CNRDataset(txt_file=os.path.join(labels_path, "val.txt"), images_path=images_path, transform=transform)

# DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# Modelo preentrenado ResNet-50
model = models.resnet50(pretrained=True)
num_classes = 2  # 0 y 1 en este caso
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Configurar función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento y validación
train_losses, val_losses = [], []
val_accuracies = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    # Entrenamiento
    model.train()
    running_loss = 0.0
    train_progress = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in train_progress:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_progress.set_postfix({"Batch Loss": loss.item()})

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validación
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        val_progress = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, labels in val_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    val_accuracies.append(accuracy)

    print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

# Métricas finales
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Empty", "Occupied"]))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Guardar el modelo entrenado
model_path = "resnet50_cnr_ext.pth"
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")

# Graficar las curvas de pérdida y precisión
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()