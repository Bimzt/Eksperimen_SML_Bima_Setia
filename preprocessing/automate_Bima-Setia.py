import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

RAW_PATH   = os.path.join(os.path.dirname(__file__), '..', 'heart_raw', 'heart.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'heart_preprocessing')

NUMERICAL_COLS   = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
OHE_COLS         = ['cp', 'restecg', 'slope', 'thal']
TARGET_COL       = 'target'


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load_data] Shape awal: {df.shape}")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[remove_duplicates] {before - len(df)} baris duplikat dihapus → {len(df)} baris")
    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERICAL_COLS:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"[handle_missing] {col}: diisi median={median:.2f}")
    for col in CATEGORICAL_COLS:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            print(f"[handle_missing] {col}: diisi modus={mode}")
    print(f"[handle_missing] Total missing: {df.isnull().sum().sum()}")
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in NUMERICAL_COLS:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        print(f"[remove_outliers] {col}: {before - len(df_clean)} baris dihapus")
    print(f"[remove_outliers] Shape setelah: {df_clean.shape}")
    return df_clean

def add_age_group(df: pd.DataFrame, df_original_age: pd.Series) -> pd.DataFrame:
    bins   = [0, 40, 50, 60, 100]
    labels = ['<40', '40-50', '50-60', '>60']
    df['age_group'] = pd.cut(df_original_age.reindex(df.index), bins=bins, labels=labels)
    le = LabelEncoder()
    df['age_group'] = le.fit_transform(df['age_group'].astype(str))
    print("[add_age_group] Binning age selesai")
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=False)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"[encode_categorical] Shape setelah OHE: {df.shape}")
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df[NUMERICAL_COLS] = scaler.fit_transform(df[NUMERICAL_COLS])
    print("[scale_features] Standarisasi selesai")
    return df

def split_and_save(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    df.to_csv(os.path.join(output_dir, 'heart_preprocessed.csv'), index=False)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    print(f"[split_and_save] X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"[split_and_save] File disimpan di: {output_dir}/")

def preprocess(raw_path: str = RAW_PATH, output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    print("AUTOMATE PREPROCESSING - Heart Disease Dataset")
    df = load_data(raw_path)
    df = remove_duplicates(df)
    df = handle_missing(df)
    age_original = df['age'].copy()
    df = remove_outliers(df)
    df = add_age_group(df, age_original)
    df = encode_categorical(df)
    df = scale_features(df)
    split_and_save(df, output_dir)
    print("\nPreprocessing selesai!")
    return df


if __name__ == '__main__':
    preprocess()
