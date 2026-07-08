import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
from logger_config import setup_logging

# Initialize Logger for Model Auditing
setup_logging()
logger = logging.getLogger("Model")

def build_and_train_hybrid_model(df, mse_threshold_pct, iso_threshold_pct, outlier_fraction):
    """
    Executes the firewall split, isolated feature processing (Log Transformation for Skew),
    Standard Scaling, Autoencoder latent feature extraction, and hybrid evaluation.
    """
    
    # -------------------------------------------------------------------------
    # 📊 PHASE 2.5: t-SNE VISUALIZATION (Purely Isolated for Analysis)
    # -------------------------------------------------------------------------
    logger.info("Starting isolated t-SNE Visualization...")
    # Creating a temporary sample to prevent heavy computation and transformation leakage on full dataset
    fraud_sample = df[df['Class'] == 1]
    normal_sample = df[df['Class'] == 0].sample(5000, random_state=42)
    df_tsne = pd.concat([fraud_sample, normal_sample])
    
    X_tsne = df_tsne.drop('Class', axis=1)
    y_tsne = df_tsne['Class']

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_transformed = tsne.fit_transform(X_tsne)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1],
                    hue=y_tsne, palette='Set1',
                    size=y_tsne, sizes=(40, 10), alpha=0.8)
    plt.title('t-SNE Visualization: Isolated Normal vs Fraud Clusters')
    plt.legend(title='Class', labels=['Normal', 'Fraud'])
    plt.show()

    # -------------------------------------------------------------------------
    # ✂️ PHASE 3: THE FIREWALL (Strict Row-Level Train/Dev/Test Split)
    # -------------------------------------------------------------------------
    logger.info("Executing The Firewall: Splitting dataset to lock Dev and Test sets...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Step 1: Isolate Test Set (20%) using stratification to preserve class distributions
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Step 2: Isolate Dev/Validation Set (20% of Train Full) to block leakage during downstream fit operations
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

    logger.info(f"Data Splits Matrix -> Train: {X_train.shape} | Val/Dev: {X_val.shape} | Test: {X_test.shape}")

    # -------------------------------------------------------------------------
    # 🛠️ PHASE 4: FEATURE ENGINEERING & PROCESSING (Post-Split Isolation)
    # -------------------------------------------------------------------------
    logger.info("Handling Skewness: Applying Log Transformation strictly after the Firewall Split...")
    
    # Avoiding SettingWithCopyWarning by explicit deep copying or direct assignment
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Mathematical Treatment for Skewed 'Amount' feature across separate matrices
    X_train['Amount'] = np.log1p(X_train['Amount'])
    X_val['Amount'] = np.log1p(X_val['Amount'])
    X_test['Amount'] = np.log1p(X_test['Amount'])
    
    logger.info("Skewness successfully treated via Log Transformation.")

    # Encoding & Scaling Template (Calculated on Train Set ONLY to prevent Statistical Leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Standard Scaling completed. Features are fully ready for Deep Learning feed.")

    # -------------------------------------------------------------------------
    # 🏗️ PHASE 5: DEEP AUTOENCODER ARCHITECTURE
    # -------------------------------------------------------------------------
    logger.info("Building Deep Autoencoder Architecture...")
    tf.keras.utils.set_random_seed(42)

    input_dim = X_train_scaled.shape[1]
    input_layer = Input(shape=(input_dim,), name="Input_Layer")

    # Symmetric Bottleneck Architecture
    encoder = Dense(128, activation="relu")(input_layer)
    encoder = Dense(64, activation="relu")(encoder)
    encoder = Dense(32, activation="relu")(encoder)
    encoder = Dense(16, activation="relu")(encoder)
    latent_space = Dense(10, activation="relu", name="Latent_Space")(encoder)

    decoder = Dense(16, activation="relu")(latent_space)
    decoder = Dense(32, activation="relu")(decoder)
    decoder = Dense(64, activation="relu")(decoder)
    decoder = Dense(128, activation="relu")(decoder)
    output_layer = Dense(input_dim, activation="linear", name="Output_Layer")(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    logger.info("Starting Autoencoder training with completely isolated Validation Set...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(X_val_scaled, X_val_scaled),  # Pure isolated validation mapping
        callbacks=[early_stop],
        verbose=1
    )

    # Extracting Latent Features Representation
    encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('Latent_Space').output)
    X_train_latent = encoder_model.predict(X_train_scaled)
    X_test_latent = encoder_model.predict(X_test_scaled)

    logger.info(f"Latent Compression Completed -> Original Dims: {X_train_scaled.shape[1]} | Latent Space: {X_train_latent.shape[1]}")

    # -------------------------------------------------------------------------
    # 🌲 PHASE 6: ISOLATION FOREST ON LATENT SPACE
    # -------------------------------------------------------------------------
    logger.info("Training Isolation Forest on Autoencoder Latent Space...")
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=outlier_fraction,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_latent)

    # Scoring Test Instances via Decision Function
    test_scores = iso_forest.decision_function(X_test_latent)
    
    # Applying the Targeted Custom Percentile Threshold to isolate anomalies
    custom_threshold = np.percentile(test_scores, iso_threshold_pct)
    custom_predictions = [1 if s <= custom_threshold else 0 for s in test_scores]

    print(f"Mean Anomaly Score: {np.mean(test_scores):.4f}")
    print(f"Score Stability (Std Dev): {np.std(test_scores):.4f}")

    plt.figure(figsize=(10, 6))
    sns.histplot(test_scores, bins=50, kde=True, color='purple')
    plt.axvline(custom_threshold, color='red', linestyle='--', label='Custom Threshold')
    plt.title('Anomaly Scores Distribution (Stability Check)')
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------
    # 🧪 PHASE 7: HYBRID EVALUATION SYSTEM (ISO OR AE MSE)
    # -------------------------------------------------------------------------
    logger.info("Executing Final Hybrid Ensembling Evaluation...")
    
    # Autoencoder Reconstruction Error Mapping
    X_test_predictions = autoencoder.predict(X_test_scaled)
    mse_scores = np.mean(np.power(X_test_scaled - X_test_predictions, 2), axis=1)
    mse_threshold = np.percentile(mse_scores, mse_threshold_pct)
    ae_predictions = [1 if e > mse_threshold else 0 for e in mse_scores]

    # Logical OR Fusion using Custom Targeted Isolation Forest Predictions
    final_hybrid_predictions = [1 if (iso == 1 or ae == 1) else 0 
                                for iso, ae in zip(custom_predictions, ae_predictions)]

    print("\n========== Hybrid System (ISO + AE MSE) Evaluation ==========")
    print(classification_report(y_test, final_hybrid_predictions))
    
    logger.info(f"Hybrid System Accuracy: {accuracy_score(y_test, final_hybrid_predictions):.4f}")
    logger.info(f"Hybrid System MSE Threshold: {mse_threshold:.4f} | ISO Threshold: {custom_threshold:.4f}")

    # Confusion Matrix Presentation
    plt.figure(figsize=(8, 6))
    cm_hybrid = confusion_matrix(y_test, final_hybrid_predictions)
    sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Reds')
    plt.title('Final Hybrid System Confusion Matrix')
    plt.show()

    error_df = pd.DataFrame({'reconstruction_error': mse_scores, 'true_class': y_test})
    logger.info("Stability Summary (MSE Error) Grouped by Target Class:")
    print(error_df.groupby('true_class')['reconstruction_error'].describe())

    # Return structural objects for downstream MLflow tracking or real-time serving pipelines
    return {
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "autoencoder": autoencoder,
        "encoder": encoder_model,
        "iso_forest": iso_forest,
        "mse_threshold": mse_threshold,
        "iso_threshold": custom_threshold,
        "final_predictions": final_hybrid_predictions
    }
