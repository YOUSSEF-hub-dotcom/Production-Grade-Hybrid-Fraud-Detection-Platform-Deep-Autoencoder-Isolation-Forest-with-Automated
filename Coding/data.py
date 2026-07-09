import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Initialize logger for data auditing
logger = logging.getLogger("Data")

def load_and_preprocess_data(file_path=r"creditcard.csv"):
    """
    Loads the dataset, performs structural inspection, executes basic cleaning, 
    and runs initial Exploratory Data Analysis (EDA) focused on distributions and imbalances.
    """
    # Corrected string formatting for logging
    logger.info(f"Loading and preprocessing data from {file_path}")

    # -------------------------------------------------------------------------
    #  PHASE 1: QUICK DATA INSPECTION
    # -------------------------------------------------------------------------
    df = pd.read_csv(file_path)
    pd.set_option('display.width', None)
    
    logger.info("Displaying first 30 rows of the dataset:")
    print(df.head(30))

    logger.info("=========== Basic Functions ==========")
    logger.info("Information about data structure and memory usage:")
    print(df.info())

    logger.info("Statistical summary of numerical features:")
    print(df.describe().round(2))

    logger.info("Dataset column names:")
    print(df.columns)

    logger.info("Dataset shape (Rows, Columns):")
    print(df.shape)

    logger.info("Column data types:")
    print(df.dtypes)

    # -------------------------------------------------------------------------
    #  PHASE 2: BASIC DATA CLEANING
    # -------------------------------------------------------------------------
    logger.info("=========== Data Cleaning ==========")
    
    # Identify and audit duplicate rows before removal
    duplicate_count = df.duplicated().sum()
    logger.info(f"Total duplicate rows identified: {duplicate_count}")
    
    df = df.drop_duplicates()
    logger.info(f"Dataset shape after removing duplicates: {df.shape}")

    # Audit missing values across all features
    logger.info('Missing values count per column:')
    print(df.isnull().sum())

    # Visual inspection of missing data distribution
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

    # -------------------------------------------------------------------------
    #  PHASE 3: EDA (Understanding Distribution, Skew, Outliers & Imbalance)
    # -------------------------------------------------------------------------
    logger.info("=========== EDA & Visualization ==========")

    # 1. Understanding Distribution & Skewness (Raw Amount)
    raw_skew_amount = df['Amount'].skew()
    logger.info(f"Raw 'Amount' feature skewness: {raw_skew_amount:.2f}")

    plt.figure(figsize=(10, 4))
    sns.histplot(df['Amount'], kde=True)
    plt.title('Raw Distribution of Amount (Before Log Transformation)')
    plt.show()

    # 2. Detecting Outliers (Raw Amount using Boxenplot)
    # NOTE: Calculated here for analytical and documentation purposes only.
    # Actual mathematical capping/transformation belongs to Phase 4 (Post-Split).
    print(df['Amount'].describe().round(2))

    q1 = df['Amount'].quantile(0.25)
    q3 = df['Amount'].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]

    logger.info(f"Outlier Detection Bounds -> Lower: {lower_bound}, Upper: {upper_bound}")
    logger.info(f"Total Outliers detected in raw 'Amount': {len(outliers)}")
    logger.info(f"Percentage of Outliers in dataset: {len(outliers)/len(df)*100:.2f}%")

    plt.figure(figsize=(10, 5))
    sns.boxenplot(x=df['Amount'])
    plt.title('Distribution of Amount (Detecting Outliers via Boxenplot)')
    plt.show()

    # [Senior Design Decision] Keeping outliers in 'Amount' column.
    # In Anomaly/Fraud Detection, extreme values are the primary signals of interest.
    # Deleting them risks removing the exact behavior the unsupervised model needs to isolate.

    # 3. Class Imbalance Check
    logger.info("Checking class imbalance for target variable:")
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud) - Log Scaled')
    plt.yscale('log')
    plt.show()

    # 4. Bivariate Analysis (Feature vs Target Distributions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.kdeplot(df[df['Class'] == 0]['Amount'], label='Normal', ax=ax1)
    sns.kdeplot(df[df['Class'] == 1]['Amount'], label='Fraud', ax=ax1)
    ax1.set_title('Amount Distribution: Normal vs Fraud')
    ax1.legend()

    sns.kdeplot(df[df['Class'] == 0]['Time'], label='Normal', ax=ax2)
    sns.kdeplot(df[df['Class'] == 1]['Time'], label='Fraud', ax=ax2)
    ax2.set_title('Time Distribution: Normal vs Fraud')
    ax2.legend()

    plt.show()

    # [Documentation] Amount Distribution Analysis:
    # Significant overlap observed. Univariate profiling is insufficient; latent feature 
    # extractions (e.g., Autoencoders) are required to capture higher-order relationships.

    # [Documentation] Time Distribution Analysis:
    # Normal traffic follows a clear diurnal/cyclic human pattern. Fraudulent traffic 
    # bypasses these cycles, showing high relative density during off-peak operational hours.

    logger.info("Data profiling and initial EDA completed successfully.")
    return df

if __name__ == "__main__":
    # Setup basic logging configuration for execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    clean_df = load_and_preprocess_data()
