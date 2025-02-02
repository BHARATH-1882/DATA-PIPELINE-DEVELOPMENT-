Iimport pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Step 1: Extract - Load dataset
file_path = "data/raw_data.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Display initial info
print("Initial Data Summary:")
print(df.info())
print(df.head())

# Step 2: Define Preprocessing Steps

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Numeric Pipeline: Handle missing values and scale features
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', StandardScaler())  # Normalize numeric data
])

# Categorical Pipeline: Handle missing values and encode categorical data
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with most frequent value
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Apply transformations
processed_data = preprocessor.fit_transform(df)

# Convert transformed data back into a DataFrame
# Get categorical feature names after encoding
encoded_cat_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)

# Create final DataFrame
final_columns = numeric_features + list(encoded_cat_features)
processed_df = pd.DataFrame(processed_data, columns=final_columns)

# Step 3: Load - Save the transformed dataset
output_path = "data/cleaned_data.csv"
processed_df.to_csv(output_path, index=False)

print(f"\nPreprocessing complete. Cleaned data saved to {output_path}")
print(processed_df.head())
