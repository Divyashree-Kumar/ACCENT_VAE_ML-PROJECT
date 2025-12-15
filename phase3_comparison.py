#!/usr/bin/env python3
"""
Phase 3: Comparative Analysis
Compares Original, Random Duplication, and VAE-Augmented Datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

# Create results directory
os.makedirs('results', exist_ok=True)

print("\n" + "="*60)
print("PHASE 3: COMPARATIVE ANALYSIS")
print("="*60)

# ===== STEP 1: LOAD DATA =====
print("\n🔄 Loading all datasets...")

# Load Original Training Data
try:
    X_train_orig = np.load('data/X_train_scaled.npy')
    y_train_orig_str = np.load('data/y_train.npy', allow_pickle=True)
    
    # Load Test Data (Must stay constant!)
    X_test = np.load('data/X_test_scaled.npy')
    y_test_str = np.load('data/y_test.npy', allow_pickle=True)

    # Load Synthetic Data
    X_synthetic = np.load('data/X_synthetic.npy')
    y_synthetic_str = np.load('data/y_synthetic.npy', allow_pickle=True)
    
    print(f"✅ Original Train: {X_train_orig.shape}")
    print(f"✅ Test Set:      {X_test.shape}")
    print(f"✅ Synthetic VAE: {X_synthetic.shape}")

except FileNotFoundError:
    print("❌ Error: Missing data files. Run Phase 1 & 2 first.")
    exit()

# ===== STEP 2: PREPARE DATASETS =====
print("\n📊 Preparing Comparison Datasets...")

# Dataset A: Original Only
X_A, y_A = X_train_orig.copy(), y_train_orig_str.copy()
print(f"   Dataset A (Original): {X_A.shape}")

# Dataset B: Original + Random Duplication (simple oversampling)
print("   Applying Random Duplication...")
X_B = X_train_orig.copy()
y_B = y_train_orig_str.copy()

# Find minority classes and duplicate samples to balance
unique_classes, counts = np.unique(y_train_orig_str, return_counts=True)
max_count = counts.max()

for class_label, count in zip(unique_classes, counts):
    if count < max_count:
        # How many more samples do we need for this class?
        needed = max_count - count
        
        # Get indices of this class
        mask = (y_train_orig_str == class_label)
        class_indices = np.where(mask)[0]
        
        # Randomly select samples to duplicate
        duplicate_indices = np.random.choice(class_indices, size=needed, replace=True)
        
        # Add them to the dataset
        X_B = np.vstack([X_B, X_train_orig[duplicate_indices]])
        y_B = np.concatenate([y_B, y_train_orig_str[duplicate_indices]])

print(f"   Dataset B (Original + Duplication): {X_B.shape}")

# Dataset C: Original + VAE
X_C = np.vstack([X_train_orig, X_synthetic])
y_C = np.concatenate([y_train_orig_str, y_synthetic_str])
print(f"   Dataset C (Original + VAE): {X_C.shape}")

# ===== STEP 3: TRAIN & EVALUATE =====
print("\n🤖 Training Models on All 3 Datasets...")
print("-" * 60)

datasets = {
    'Original Only': (X_A, y_A),
    'Original + Duplication': (X_B, y_B),
    'Original + VAE': (X_C, y_C)
}

models = {
    'SVM': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results_list = []

for dataset_name, (X_train, y_train) in datasets.items():
    print(f"\n{dataset_name} ({X_train.shape[0]} samples)")
    print("-" * 40)
    
    for model_name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict on CONSTANT Test Set
        y_pred = model.predict(X_test)
        
        # Evaluate
        acc = accuracy_score(y_test_str, y_pred)
        f1 = f1_score(y_test_str, y_pred, average='weighted')
        
        results_list.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Accuracy': acc,
            'F1-Score': f1
        })
        print(f"   {model_name:15s} | Acc: {acc:.4f} | F1: {f1:.4f}")

# ===== STEP 4: VISUALIZE RESULTS =====
print("\n📊 Creating Comparison Plots...")

results_df = pd.DataFrame(results_list)

# Plot 1: Accuracy Comparison
fig, ax = plt.subplots(figsize=(10, 6))
pivot_acc = results_df.pivot(index='Model', columns='Dataset', values='Accuracy')
pivot_acc.plot(kind='bar', ax=ax, width=0.8)
plt.title('Accuracy Comparison: Original vs Duplication vs VAE', fontsize=14, fontweight='bold')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.6, 0.9)
plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('results/06_final_comparison_accuracy.png', dpi=150, bbox_inches='tight')
print("✅ Saved: results/06_final_comparison_accuracy.png")
plt.close()

# Plot 2: F1-Score Comparison
fig, ax = plt.subplots(figsize=(10, 6))
pivot_f1 = results_df.pivot(index='Model', columns='Dataset', values='F1-Score')
pivot_f1.plot(kind='bar', ax=ax, width=0.8)
plt.title('F1-Score Comparison: Original vs Duplication vs VAE', fontsize=14, fontweight='bold')
plt.xlabel('Model')
plt.ylabel('F1-Score')
plt.ylim(0.6, 0.9)
plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('results/07_final_comparison_f1.png', dpi=150, bbox_inches='tight')
print("✅ Saved: results/07_final_comparison_f1.png")
plt.close()

# ===== STEP 5: IMPROVEMENT ANALYSIS =====
print("\n" + "="*60)
print("📈 IMPROVEMENT ANALYSIS")
print("="*60)

# Calculate improvement for each model
improvement_data = []
for model_name in models.keys():
    orig_acc = results_df[(results_df['Dataset'] == 'Original Only') & 
                          (results_df['Model'] == model_name)]['Accuracy'].values[0]
    vae_acc = results_df[(results_df['Dataset'] == 'Original + VAE') & 
                         (results_df['Model'] == model_name)]['Accuracy'].values[0]
    
    improvement = vae_acc - orig_acc
    improvement_pct = (improvement / orig_acc) * 100
    
    improvement_data.append({
        'Model': model_name,
        'Original': orig_acc,
        'VAE': vae_acc,
        'Improvement': improvement,
        'Improvement %': improvement_pct
    })
    
    status = "✅" if improvement > 0 else "⚠️"
    print(f"{status} {model_name:15s}: {orig_acc:.4f} → {vae_acc:.4f} | Δ {improvement:+.4f} ({improvement_pct:+.2f}%)")

improvement_df = pd.DataFrame(improvement_data)
improvement_df.to_csv('results/improvement_analysis.csv', index=False)
print("\n✅ Saved: results/improvement_analysis.csv")

# ===== STEP 6: SAVE RESULTS TABLE =====
results_df.to_csv('results/final_results_table.csv', index=False)
print("✅ Saved: results/final_results_table.csv")

# ===== FINAL SUMMARY =====
print("\n" + "="*60)
print("🏆 FINAL RESULTS SUMMARY")
print("="*60)

pivot_summary = results_df.pivot(index='Model', columns='Dataset', values='Accuracy')
print("\n" + str(pivot_summary))

print("\n" + "="*60)
print("✅ PROJECT COMPLETE!")
print("="*60)
print("\n📁 Results saved to results/ folder")
print("🎉 Compare the accuracy numbers above to see if VAE helped!")
