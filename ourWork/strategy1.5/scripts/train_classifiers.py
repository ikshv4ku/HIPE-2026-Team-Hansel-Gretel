import pickle
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging

# Setup lightweight logging to hide verbose LightGBM warnings
logging.basicConfig(level=logging.INFO)

def load_all_features(pkl_files):
    X = []
    y_at = []
    y_isAt = []
    weights = []
    
    for f in pkl_files:
        if not os.path.exists(f):
            print(f"File not found, skipping: {f}")
            continue
        with open(f, 'rb') as fp:
            data = pickle.load(fp)
            X.append(data["features"])
            y_at.append(data["at_labels"])
            y_isAt.append(data["isAt_labels"])
            weights.append(data["weights"])
            
    if not X:
        raise ValueError("No valid feature files found!")
        
    X = np.concatenate(X, axis=0)
    y_at = np.concatenate(y_at, axis=0)
    y_isAt = np.concatenate(y_isAt, axis=0)
    weights = np.concatenate(weights, axis=0)
    
    return X, y_at, y_isAt, weights

def train_models(feature_files, output_dir):
    print(f"Loading features from {len(feature_files)} files...")
    X, y_at, y_isAt, weights = load_all_features(feature_files)
    
    print(f"Total training samples: {X.shape[0]}")
    
    # Train Random Forest for 'at' relation
    # at targets: 0=FALSE, 1=PROBABLE, 2=TRUE
    print("\n[1/2] Training Random Forest for 'at' relation...")
    rf_at = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_at.fit(X, y_at, sample_weight=weights)
    
    preds_at = rf_at.predict(X)
    print("Training Accuracy for 'at':", accuracy_score(y_at, preds_at))
    print(classification_report(y_at, preds_at, zero_division=0))
    
    # Train LightGBM for 'isAt' relation
    # isAt targets: 0=FALSE, 1=TRUE
    print("\n[2/2] Training LightGBM for 'isAt' relation...")
    
    import warnings
    warnings.filterwarnings('ignore') # lightgbm logs can be incredibly noisy
    
    lgbm_isAt = LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbose=-1)
    lgbm_isAt.fit(X, y_isAt, sample_weight=weights)
    
    preds_isAt = lgbm_isAt.predict(X)
    print("Training Accuracy for 'isAt':", accuracy_score(y_isAt, preds_isAt))
    print(classification_report(y_isAt, preds_isAt, zero_division=0))

    # Save models
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(rf_at, os.path.join(output_dir, "rf_at_model.joblib"))
    joblib.dump(lgbm_isAt, os.path.join(output_dir, "lgbm_isAt_model.joblib"))
    print(f"\nDone! Saved both trained models to {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/results/embeddings/")
    parser.add_argument("--output_dir", type=str, default="/home/pradyuman/IITR/sem6/lbp/model/HIPE-2026-Team-Hansel-Gretel/results/models/")
    args = parser.parse_args()
    
    # Load all available training embeddings automatically
    import glob
    train_files = glob.glob(os.path.join(args.features_dir, "train_*_features.pkl"))
    
    if len(train_files) == 0:
        print(f"Could not find any feature files in {args.features_dir}")
    else:
        train_models(train_files, args.output_dir)
