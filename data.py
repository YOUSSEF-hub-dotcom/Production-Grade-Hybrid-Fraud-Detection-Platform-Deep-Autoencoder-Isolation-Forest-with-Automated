import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path=r"creditcard.csv"):
    logger.info(f"Loading and preprocessing data from %s {file_path}")

    df = pd.read_csv(file_path)
    pd.set_option('display.width', None)
    print(df.head(30))

    logger.info("=========== Basic Functions ==========")
    logger.info("information about data:")
    print(df.info())

    logger.info("Statistical Operations:")
    print(df.describe().round(2))

    logger.info("Columns:")
    print(df.columns)

    logger.info("number of rows & columns:")
    print(df.shape)

    logger.info("Column types:")
    print(df.dtypes)

    logger.info("=========== Data Cleaning ==========")
    logger.info("Duplicate values:")
    logger.info(df.duplicated().sum()) # 1081
    print(df.shape)

    df = df.drop_duplicates()
    logger.info(f"After Removing Duplicate{df.shape}")

    logger.info('Missing Values:')
    print(df.isnull().sum())

    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Count')
    plt.show()

    logger.info("=========== Data Preprocessing ==========")

    # Skew value Amount
    skew_value = df['Amount'].skew()
    logger.info(f"Skewness before transformation: {skew_value}")

    sns.histplot(df['Amount'], kde=True)
    plt.title('Distribution of Amount Before Treatment skew')
    plt.show()

    df['Amount'] = np.log1p(df['Amount'])

    treat_skew_Amount = df['Amount'].skew()
    logger.info(f"Skewness after Log Transformation: {treat_skew_Amount}")

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

    logger.info(f"Lower Bound: {lower_bound}")
    logger.info(f"Upper Bound: {upper_bound}")
    logger.info(f"Total Outliers in Amount: {len(outliers)}")
    logger.info(f"Percentage of Outliers: {len(outliers)/len(df)*100:.2f}%")

    plt.figure(figsize=(10, 5))
    sns.boxenplot(x=df['Amount'])
    plt.title('Distribution of Amount (Detecting Outliers)')
    plt.show()

    # [Decision] Keeping outliers in 'Amount' column after transformation.
    # In Anomaly Detection, extreme values (outliers) are often the signals of potential fraud.
    # Removing them would mean removing the very instances we want the model to detect.
    # We handled the skewness (16.9 -> 0.16) to ensure model stability while preserving the anomalies.

    logger.info("=========== EDA & Visualization ==========")

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
    
    logger.info("Data preprocessing and EDA completed successfully.")
    return df

if __name__ == "__main__":
    clean_df = load_and_preprocess_data()