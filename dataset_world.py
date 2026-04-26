import kagglehub

# Download latest version
path = kagglehub.dataset_download("simongraves/license-plate-dataset")

print("Path to dataset files:", path)