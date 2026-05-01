from huggingface_hub import HfApi, create_repo
import os
import shutil

# Define your Hugging Face username and space name
hf_username = "pjhansi2404"  # IMPORTANT: Replace with your actual Hugging Face username
sapce_name = "superkart_sales_forecast_app" # Your desired space name
repo_id = f"{hf_username}/{sapce_name}"

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# 1. Create a new Hugging Face Space (if it doesn't exist)
try:
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Hugging Face Space '{repo_id}' already exists. Updating it.")
except Exception:
    print(f"Hugging Face Space '{repo_id}' not found. Creating a new one...")
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", private=False)
    print(f"Hugging Face Space '{repo_id}' created.")

# 2. Prepare local directory for upload
local_app_path = "./hf_space_app_temp"
os.makedirs(local_app_path, exist_ok=True)

# Copy deployment files into the temporary directory
shutil.copy("superkart_project/model_deployment/Dockerfile", os.path.join(local_app_path, "Dockerfile"))
shutil.copy("superkart_project/model_deployment/requirements.txt", os.path.join(local_app_path, "requirements.txt"))
shutil.copy("superkart_project/model_deployment/app.py", os.path.join(local_app_path, "app.py"))
shutil.copy("superkart_project/model_deployment/best_random_forest_model.joblib", os.path.join(local_app_path, "best_random_forest_model.joblib"))

# 3. Upload the entire folder to the Hugging Face Space
api.upload_folder(
    folder_path=local_app_path,
    repo_id=repo_id,
    repo_type="space",
    commit_message="Add Dockerfile, requirements, app.py, and model for deployment"
)

# 4. Clean up the temporary directory
shutil.rmtree(local_app_path)

print(f"Deployment files successfully pushed to Hugging Face Space: https://huggingface.co/spaces/{repo_id}")
