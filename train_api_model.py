import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load data
X_train = np.load("data/X_train_scaled.npy")
y_train = np.load("data/y_train.npy", allow_pickle=True)
X_synth = np.load("data/X_synthetic.npy")
y_synth = np.load("data/y_synthetic.npy", allow_pickle=True)

# 2. Combine Original + VAE data
X_vae_train = np.vstack([X_train, X_synth])
y_vae_train = np.concatenate([y_train, y_synth])

# 3. Scale features
scaler = StandardScaler()
X_vae_train_scaled = scaler.fit_transform(X_vae_train)

# 4. Train SVM with probabilities
svm_vae = SVC(kernel="rbf", probability=True, random_state=42)
svm_vae.fit(X_vae_train_scaled, y_vae_train)

# 5. Save model and scaler
joblib.dump(svm_vae, "vae_accent_model.pkl")
joblib.dump(scaler, "vae_scaler.pkl")

print("saved vae_accent_model.pkl and vae_scaler.pkl")
