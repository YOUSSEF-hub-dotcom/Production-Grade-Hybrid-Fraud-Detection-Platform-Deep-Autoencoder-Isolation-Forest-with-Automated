import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"creditcard.csv")
pd.set_option('display.width', None)
print(df.head(30))

print("=========== Basic Functions ==========")
print("information about data:")
print(df.info())

print("Statistical Operations:")
print(df.describe().round(2))

print("Columns:")
print(df.columns)

print("number of rows & columns:")
print(df.shape)

print("Column types:")
print(df.dtypes)

print("=========== Data Cleaning ==========")
print("Duplicate values:")
print(df.duplicated().sum()) # 1081
print(df.shape)

df = df.drop_duplicates(df)
print(f"After Removing Duplicate{df.shape}")

print('Missing Values:')
print(df.isnull().sum())

sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Count')
plt.show()

print("=========== Data Preprocessing ==========")

# Skew value Amount
skew_value = df['Amount'].skew()
print(f"skew values: {skew_value}")

sns.histplot(df['Amount'], kde=True)
plt.title('Distribution of Amount Before Treatment skew')
plt.show()

df['Amount'] = np.log1p(df['Amount'])

treat_skew_Amount = df['Amount'].skew()
print(f"Treatment Skew of Amount :{ treat_skew_Amount}")

sns.histplot(df['Amount'], kde=True)
plt.title("Distribution of Amount After Treatment Skew (Log Transformation)")
plt.show()

# Check outlier in Amount
print(df['Amount'].describe().round(2))

Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]

print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")
print(f"Total Outliers in Amount: {len(outliers)}")
print(f"Percentage of Outliers: {len(outliers)/len(df)*100:.2f}%")

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Amount'])
plt.title('Distribution of Amount (Detecting Outliers)')
plt.show()

# [Decision] Keeping outliers in 'Amount' column after transformation.
# In Anomaly Detection, extreme values (outliers) are often the signals of potential fraud.
# Removing them would mean removing the very instances we want the model to detect.
# We handled the skewness (16.9 -> 0.16) to ensure model stability while preserving the anomalies.

print("=========== EDA & Visualization ==========")

sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Normal, 1: Fraud)')
plt.yscale('log')
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

sns.kdeplot(df[df['Class'] == 0]['Amount'], label='Normal', ax=ax1)
sns.kdeplot(df[df['Class'] == 1]['Amount'], label='Fraud', ax=ax1)
ax1.set_title('Amount Distribution: Normal vs Fraud')

sns.kdeplot(df[df['Class'] == 0]['Time'], label='Normal', ax=ax2)
sns.kdeplot(df[df['Class'] == 1]['Time'], label='Fraud', ax=ax2)
ax2.set_title('Time Distribution: Normal vs Fraud')

plt.legend()
plt.show()

# [Documentation] Amount Distribution Analysis:
# - Purpose: Visualize the 'Amount' feature after Log Transformation.
# - Observation: The Fraud (Orange) and Normal (Blue) distributions overlap significantly.
# - Insight: Fraudsters often mimic typical transaction amounts to avoid detection.
# - Conclusion: Univariate analysis isn't enough; we need a Latent Representation (Autoencoder)
#   to capture hidden patterns across multiple features.

# [Documentation] Time Distribution Analysis:
# - Purpose: Examine transaction frequency over time for both classes.
# - Observation: Normal transactions show clear cyclic/diurnal patterns (peaks and troughs).
# - Insight: Fraudulent activities (Orange) often deviate from these natural human cycles,
#   showing activity during typical 'low-volume' periods.
# - Conclusion: 'Time' is a critical behavioral feature for Anomaly Detection.

# The 'Class' labels used here are for EDA/Documentation purposes ONLY.
# They will be COMPLETELY dropped during the model training phase to maintain
# the Unsupervised integrity of the system.

print("=========== Building Model ==========")
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest


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


X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed. Data is now ready for the Autoencoder.")


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

encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('Latent_Space').output)

X_train_latent = encoder_model.predict(X_train_scaled)
X_test_latent = encoder_model.predict(X_test_scaled)

print(f"Original shape: {X_train_scaled.shape}")
print(f"Latent shape: {X_train_latent.shape}")


outlier_fraction = 0.02  # 2%

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

custom_threshold = np.percentile(test_scores, 0.18)
custom_predictions = [1 if s <= custom_threshold else 0 for s in test_scores]


print(f"Mean Anomaly Score: {np.mean(test_scores):.4f}")
print(f"Score Stability (Std Dev): {np.std(test_scores):.4f}")

plt.figure(figsize=(10, 6))
sns.histplot(test_scores, bins=50, kde=True, color='purple')
plt.axvline(custom_threshold, color='red', linestyle='--', label='Threshold')
plt.title('Anomaly Scores Distribution (Stability Check)')
plt.legend()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("========== Final Model Evaluation ==========")
print(f"Accuracy Score: {accuracy_score(y_test, test_predictions_cleaned):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_predictions_cleaned))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions_cleaned))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, test_predictions_cleaned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Hybrid AE-ISO Confusion Matrix')
plt.show()


X_test_predictions = autoencoder.predict(X_test_scaled)
mse_scores = np.mean(np.power(X_test_scaled - X_test_predictions, 2), axis=1)

mse_threshold = np.percentile(mse_scores, 99)
ae_predictions = [1 if e > mse_threshold else 0 for e in mse_scores]


final_hybrid_predictions = [1 if (iso == 1 or ae == 1) else 0
                            for iso, ae in zip(test_predictions_cleaned, ae_predictions)]

print("========== Hybrid System (ISO + AE MSE) Evaluation ==========")

print(classification_report(y_test, final_hybrid_predictions))

plt.figure(figsize=(8, 6))
cm_hybrid = confusion_matrix(y_test, final_hybrid_predictions)
sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Reds')
plt.title('Final Hybrid System Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

error_df = pd.DataFrame({'reconstruction_error': mse_scores, 'true_class': y_test})
print("\nStability Summary (MSE):")
print(error_df.groupby('true_class')['reconstruction_error'].describe())
