import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from autoencoder import *
from help_func import *

# Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">> My model using", device)

# Initialize model
model = ResNetAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30

# Training with better progress tracking
train_losses = []
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    for data, _ in progress_bar:
        data = data.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    
    # Save best model
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), 'best_autoencoder.pth')
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}')

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, 'b-o', linewidth=2, markersize=8)
plt.title('Training Loss Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Load best model
model.load_state_dict(torch.load('best_autoencoder.pth'))
model.eval()

# Evaluation with better metrics
def evaluate_model(model, normal_loader, anomalous_data):
    # Calculate reconstruction errors
    normal_errors = []
    for data, _ in normal_loader:
        data = data.to(device)
        with torch.no_grad():
            recon = model(data)
        error = torch.mean((recon - data)**2, dim=(1, 2, 3)).cpu().numpy()
        normal_errors.extend(error)
    
    anomalous_data = anomalous_data.to(device)
    with torch.no_grad():
        anomalous_recon = model(anomalous_data)
    anomalous_errors = torch.mean((anomalous_recon - anomalous_data)**2, dim=(1, 2, 3)).cpu().numpy()
    
    return np.array(normal_errors), np.array(anomalous_errors)

# Get evaluation data
normal_data, _ = next(iter(train_loader))
anomalous_data = torch.randn_like(normal_data) * 0.25  # Synthetic anomalies

normal_errors, anomalous_errors = evaluate_model(model, train_loader, anomalous_data)

# Calculate optimal threshold (Youden's J statistic)
labels = np.concatenate([np.zeros_like(normal_errors), np.ones_like(anomalous_errors)])
scores = np.concatenate([normal_errors, anomalous_errors])
fpr, tpr, thresholds = roc_curve(labels, scores)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Make predictions
predictions = (scores > optimal_threshold).astype(int)

# Enhanced visualization
plt.figure(figsize=(15, 10))

# Histogram plot
plt.subplot(2, 2, 1)
plt.hist(normal_errors, bins=50, alpha=0.7, color='blue', label='Normal')
plt.hist(anomalous_errors, bins=50, alpha=0.7, color='red', label='Anomalous')
plt.axvline(optimal_threshold, color='k', linestyle='--', label=f'Threshold: {optimal_threshold:.2f}')
plt.title('Reconstruction Error Distribution', fontsize=14)
plt.xlabel('Reconstruction Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# ROC Curve
plt.subplot(2, 2, 2)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black', label='Optimal Threshold')
plt.title('ROC Curve', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)

# Confusion Matrix
plt.subplot(2, 2, 3)
cm = confusion_matrix(labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomalous'], 
            yticklabels=['Normal', 'Anomalous'])
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)

plt.tight_layout()
plt.show()

# Print comprehensive metrics
print("\n" + "="*60)
print(" " * 20 + "EVALUATION METRICS")
print("="*60)
print(f"{'Mean Normal Error:':<30}{np.mean(normal_errors):.4f}")
print(f"{'Mean Anomalous Error:':<30}{np.mean(anomalous_errors):.4f}")
print(f"{'Optimal Threshold:':<30}{optimal_threshold:.4f}")
print(f"{'ROC AUC Score:':<30}{roc_auc:.4f}")
print("\n" + "-"*60)
print(classification_report(labels, predictions, target_names=['Normal', 'Anomalous']))
print("="*60 + "\n")

# Visualization of reconstructed images
plot_reconstructed_images(model, normal_data[:8], title='Normal Samples Reconstruction')
plot_reconstructed_images(model, anomalous_data[:8], title='Anomalous Samples Reconstruction')

# Latent space visualization if available
if 'plot_latent_space' in globals():
    plot_latent_space(normal_data, model, title='Latent Space - Normal Data')
    plot_latent_space(anomalous_data, model, title='Latent Space - Anomalous Data')
