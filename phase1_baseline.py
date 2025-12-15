#!/usr/bin/env python3
"""
Phase 1: Baseline Models on Original Dataset
FINAL VERSION - Loads from local CSV file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

print("\n" + "="*60)
print("PHASE 1: BASELINE MODELS")
print("="*60)

# ===== STEP 1: LOAD DATASET =====
print("\n🔄 Loading Speaker Accent Recognition Dataset...")

# Path to the file you manually downloaded
file_path = 'data/accent-mfcc-data.csv'

if not os.path.exists(file_path):
    print(f"❌ Error: File not found at {file_path}")
    print("   Please download 'accent-mfcc-data.csv' and place it in the 'data' folder.")
    exit()

# Load CSV (header=0 means first row is header)
df = pd.read_csv(file_path)

print(f"✅ Dataset Loaded Successfully!")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# ===== STEP 2: PREPARE DATA =====
print("\n📊 Preparing data...")

# The target column is 'language'
target_col = 'language'

# Separate features (X) and target (y)
X = df.drop(columns=[target_col]).values
y = df[target_col].values

print(f"   Total Samples: {X.shape[0]}")
print(f"   Features: {X.shape[1]}")
print(f"   Unique Classes: {len(np.unique(y))}")

# Check class distribution
unique_classes, counts = np.unique(y, return_counts=True)
print(f"\n   Class Distribution:")
for label, count in zip(unique_classes, counts):
    print(f"   {label}: {count}")

# ===== STEP 3: SPLIT DATA =====
print(f"\n📊 Splitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

# ===== STEP 4: NORMALIZE DATA =====
print(f"\n🔧 Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== STEP 5: TRAIN MODELS =====
print(f"\n🤖 Training baseline models...")
print("-" * 60)

baseline_results = {}

# SVM
print("\n1️⃣  SVM (Support Vector Machine)")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
baseline_results['SVM'] = {'accuracy': svm_acc, 'f1': svm_f1}
print(f"   ✅ Accuracy: {svm_acc:.4f}")
print(f"   ✅ F1-Score: {svm_f1:.4f}")

# Random Forest
print("\n2️⃣  Random Forest Classifier")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
baseline_results['Random Forest'] = {'accuracy': rf_acc, 'f1': rf_f1}
print(f"   ✅ Accuracy: {rf_acc:.4f}")
print(f"   ✅ F1-Score: {rf_f1:.4f}")

# k-NN
print("\n3️⃣  k-Nearest Neighbors (k=5)")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred, average='weighted')
baseline_results['KNN'] = {'accuracy': knn_acc, 'f1': knn_f1}
print(f"   ✅ Accuracy: {knn_acc:.4f}")
print(f"   ✅ F1-Score: {knn_f1:.4f}")

# ===== STEP 6: VISUALIZE RESULTS =====
print("\n📊 Creating visualizations...")

models = list(baseline_results.keys())
accuracies = [baseline_results[m]['accuracy'] for m in models]
f1_scores = [baseline_results[m]['f1'] for m in models]

# Bar plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Baseline Models - Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 1)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

axes[1].bar(models, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[1].set_ylabel('F1-Score', fontsize=12)
axes[1].set_title('Baseline Models - F1-Score Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 1)
for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/02_baseline_comparison.png', dpi=150)
print("   ✅ Saved: results/02_baseline_comparison.png")
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=unique_classes, yticklabels=unique_classes)
plt.title('SVM - Confusion Matrix (Original Data)', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/03_confusion_matrix_baseline.png', dpi=150)
print("   ✅ Saved: results/03_confusion_matrix_baseline.png")
plt.close()

# Save to CSV
results_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'F1-Score': f1_scores
})
results_df.to_csv('results/baseline_results.csv', index=False)
print("   ✅ Saved: results/baseline_results.csv")

# Save processed data for next steps
np.save('data/X_train_scaled.npy', X_train_scaled)
np.save('data/X_test_scaled.npy', X_test_scaled)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)
print("   ✅ Saved: data/X_train_scaled.npy, y_train.npy")

print("\n" + "="*60)
print("✅ PHASE 1 COMPLETE!")
print("="*60)
