import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# Create results directory if needed
os.makedirs('results', exist_ok=True)

print("\n" + "="*60)
print("PHASE 2: VARIATIONAL AUTOENCODER (VAE)")
print("="*60)

# ===== STEP 1: LOAD PROCESSED DATA =====
print("\n🔄 Loading data from Phase 1...")

try:
    X_train = np.load('data/X_train_scaled.npy')
    y_train_str = np.load('data/y_train.npy', allow_pickle=True) # Allow string loading
    print(f"✅ Loaded X_train: {X_train.shape}")
    print(f"✅ Loaded y_train: {y_train_str.shape}")
except FileNotFoundError:
    print("❌ Error: Processed data not found. Run phase1_baseline.py first!")
    exit()

# Convert string labels ('ES', 'FR', etc.) to numbers (0, 1, 2...)
unique_labels = np.unique(y_train_str)
label_map = {label: i for i, label in enumerate(unique_labels)}
y_train = np.array([label_map[label] for label in y_train_str])

print("\n✅ Converted string labels to numbers")
print(f"   Mapping: {label_map}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ===== STEP 2: DEFINE VAE ARCHITECTURE =====
class VAE(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=8, latent_dim=2):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss_function(x_recon, x, mu, logvar):
    recon_loss = nn.MSELoss(reduction='sum')(x_recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

print("\n✅ VAE Model Architecture Defined")
print("   Input: 12 -> Hidden: 8 -> Latent: 2")

# ===== STEP 3: TRAIN THE VAE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Training on: {device}")

vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

num_epochs = 100
print(f"\n🚀 Starting Training ({num_epochs} Epochs)...")
print("-" * 60)

losses = []
for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, _ in train_loader:
        batch_X = batch_X.to(device)
        
        x_recon, mu, logvar = vae(batch_X)
        loss = vae_loss_function(x_recon, batch_X, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(X_train)
    losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

print("-" * 60)
print("✅ Training Complete!")

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(losses, color='#4ECDC4')
plt.title('VAE Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.savefig('results/04_vae_training_loss.png', dpi=150)
print("\n✅ Saved: results/04_vae_training_loss.png")
plt.close()

# ===== STEP 4: VISUALIZE LATENT SPACE =====
print("\n🔍 Visualizing Latent Space...")

vae.eval()
with torch.no_grad():
    _, mu, _ = vae(X_train_tensor.to(device))
    latent_points = mu.cpu().numpy()

plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('viridis', len(unique_labels))

for i, label in enumerate(unique_labels):
    mask = (y_train == i)
    plt.scatter(latent_points[mask, 0], latent_points[mask, 1], 
               label=label, alpha=0.7, s=80, color=colors(i))

plt.title('VAE Latent Space: Accent Clusters')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/05_latent_space_visualization.png', dpi=150)
print("✅ Saved: results/05_latent_space_visualization.png")
plt.close()

# ===== STEP 5: GENERATE SYNTHETIC DATA =====
print("\n🔧 Generating Synthetic Data...")

# Generate 50 new samples for each accent
samples_per_accent = 50
X_synthetic_list = []
y_synthetic_list = []

vae.eval()
with torch.no_grad():
    for i in range(len(unique_labels)):
        # Sample points from a standard normal distribution
        z_samples = torch.randn(samples_per_accent, 2).to(device)
        
        # Decode to get synthetic features
        x_generated = vae.decoder(z_samples)
        
        X_synthetic_list.append(x_generated.cpu().numpy())
        y_synthetic_list.extend([i] * samples_per_accent)

X_synthetic = np.vstack(X_synthetic_list)
y_synthetic_str = np.array([unique_labels[i] for i in y_synthetic_list])

print(f"✅ Generated {X_synthetic.shape[0]} synthetic samples.")

# Save synthetic data
np.save('data/X_synthetic.npy', X_synthetic)
np.save('data/y_synthetic.npy', y_synthetic_str)
print("✅ Saved: data/X_synthetic.npy, data/y_synthetic.npy")

print("\n" + "="*60)
print("✅ PHASE 2 COMPLETE!")
print("🚀 Next Step: Run python phase3_comparison.py")
print("="*60)
