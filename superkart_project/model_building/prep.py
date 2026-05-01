# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

# IMPORTANT: Replace 'YOUR_HF_USERNAME' with your actual Hugging Face username
hf_username = "pjhansi2404"
repo_id = f"{hf_username}/superrkart"

DATASET_PATH = f"hf://datasets/pjhansi2404/superrkart/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifiers that are not useful as features
df.drop(columns=['Product_Id', 'Store_Id'], inplace=True)

# Encoding categorical columns from the data description
label_encoder = LabelEncoder()
categorical_cols = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])
    else:
        print(f"Warning: Categorical column '{col}' not found for encoding.")

# Set the target variable as identified from the problem statement
target_col = 'Product_Store_Sales_Total'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type="dataset",
    )
