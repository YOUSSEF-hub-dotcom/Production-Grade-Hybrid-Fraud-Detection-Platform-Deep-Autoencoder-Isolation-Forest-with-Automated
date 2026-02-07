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


# إعداد الـ Logger
setup_logging()
logger = logging.getLogger(__name__)


def build_and_train_hybrid_model(df, mse_threshold_pct, iso_threshold_pct, outlier_fraction):
    ## 1. t-SNE Visualization
    # ======================================================
    logger.info("Starting t-SNE Visualization...")
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0].sample(5000, random_state=42)

    df_tsne = pd.concat([fraud, normal])
    X_tsne = df_tsne.drop('Class', axis=1)
    y_tsne = df_tsne['Class']

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_transformed = tsne.fit_transform(X_tsne)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1],
                    hue=y_tsne, palette='Set1',
                    size=y_tsne, sizes=(40, 10),
                    alpha=0.8)
    plt.title('t-SNE Visualization: Normal vs Fraud Clusters')
    plt.legend(title='Class', labels=['Normal', 'Fraud'])
    plt.show()

    # [Documentation] t-SNE High-Dimensional Mapping:
    # - Purpose: Visualize how 30 features separate in 2D space.
    # - Observation: Fraud instances (Red) form distinct, string-like manifolds (curves).
    # - Insight: Fraud is not randomly scattered; it has a specific "Geometric Signature".
    # - Conclusion: This high separability ensures that the Isolation Forest will
    #   be highly effective at isolating these red threads from the normal data bulk.

    ## 2. Data Splitting & Scaling
    # ======================================================
    logger.info("Splitting and Scaling data...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Scaling completed. Data is now ready for the Autoencoder.")

    ## 3. Building Autoencoder Architecture
    # ======================================================
    logger.info("Building Autoencoder Architecture...")

    tf.keras.utils.set_random_seed(42)

    input_dim = X_train_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))

    encoder = Dense(128, activation="relu")(input_layer)
    encoder = Dense(64, activation="relu")(encoder)
    encoder = Dense(32, activation="relu")(encoder)
    encoder = Dense(16, activation="relu")(encoder)

    latent_space = Dense(10, activation="relu", name="Latent_Space")(encoder)

    decoder = Dense(16, activation="relu")(latent_space)
    decoder = Dense(32, activation="relu")(decoder)
    decoder = Dense(64, activation="relu")(decoder)
    decoder = Dense(128, activation="relu")(decoder)

    output_layer = Dense(input_dim, activation="linear")(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    ## 4. Training Autoencoder
    # ======================================================
    logger.info("Starting Autoencoder training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Extraction of Encoder for latent features
    encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('Latent_Space').output)
    X_train_latent = encoder_model.predict(X_train_scaled)
    X_test_latent = encoder_model.predict(X_test_scaled)

    logger.info(f"Original shape: {X_train_scaled.shape} | Latent shape: {X_train_latent.shape}")

    ## 5. Isolation Forest on Latent Features
    # ======================================================
    logger.info("Training Isolation Forest on Latent Space...")
    
    #outlier_fraction = 0.02 # 2%

    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=outlier_fraction,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_latent)


    test_predictions = iso_forest.predict(X_test_latent)

    test_predictions_cleaned = [1 if x == -1 else 0 for x in test_predictions]

    test_scores = iso_forest.decision_function(X_test_latent)

    custom_threshold = np.percentile(test_scores, iso_threshold_pct)
    custom_predictions = [1 if s <= custom_threshold else 0 for s in test_scores]


    print(f"Mean Anomaly Score: {np.mean(test_scores):.4f}")
    print(f"Score Stability (Std Dev): {np.std(test_scores):.4f}")

    plt.figure(figsize=(10, 6))
    sns.histplot(test_scores, bins=50, kde=True, color='purple')
    plt.axvline(custom_threshold, color='red', linestyle='--', label='Threshold')
    plt.title('Anomaly Scores Distribution (Stability Check)')
    plt.legend()
    plt.show()
    ## 6. Evaluation (Hybrid System)
    # ======================================================
    logger.info("Final Model Evaluation...")
    
    # AE MSE Predictions
    X_test_predictions = autoencoder.predict(X_test_scaled)
    mse_scores = np.mean(np.power(X_test_scaled - X_test_predictions, 2), axis=1)
    mse_threshold = np.percentile(mse_scores, mse_threshold_pct)
    ae_predictions = [1 if e > mse_threshold else 0 for e in mse_scores]

    # Hybrid Logic (ISO OR AE)
    final_hybrid_predictions = [1 if (iso == 1 or ae == 1) else 0 
                                for iso, ae in zip(test_predictions_cleaned, ae_predictions)]

    print("\n========== Hybrid System (ISO + AE MSE) Evaluation ==========")
    logger.info(classification_report(y_test, final_hybrid_predictions))
    logger.info(f"Hybrid System Accuracy: {accuracy_score(y_test, final_hybrid_predictions):.4f}")
    logger.info(f"Hybrid System MSE Threshold: {mse_threshold:.4f} | ISO Threshold: {custom_threshold:.4f}")

    plt.figure(figsize=(8, 6))
    cm_hybrid = confusion_matrix(y_test, final_hybrid_predictions)
    sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Reds')
    plt.title('Final Hybrid System Confusion Matrix')
    plt.show()

    error_df = pd.DataFrame({'reconstruction_error': mse_scores, 'true_class': y_test})
    logger.info("\nStability Summary (MSE):")
    logger.info(error_df.groupby('true_class')['reconstruction_error'].describe())

    # إرجاع كل العناصر المهمة للـ MLOps
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