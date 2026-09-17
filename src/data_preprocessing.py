import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(df, target_col='label'):
    """
    Preprocess the dataset including:
    - Handling missing values
    - Encoding categorical variables
    - Train-test split
    - Feature scaling
    """
    
    # 1. Drop irrelevant columns such as row indices if they exist
    if 'row' in df.columns:
        df = df.drop(columns=['row'])
        
    # 2. Handle missing values
    # For binary/categorical symptom data, filling with mode is a reasonable approach
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            
    # 3. Encode categorical features (if any are not already binary)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
    # 4. Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        
    X = df.drop(columns=[target_col])
    # Keep strictly the first 58 symptom columns as requested
    X = X.iloc[:, :58]
    y = df[target_col]
    
    # Ensure all columns are converted to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    # Fill any new NaNs resulting from coercion with 0 (default absence of symptom)
    X = X.fillna(0)
    
    # 5. Train-test split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
