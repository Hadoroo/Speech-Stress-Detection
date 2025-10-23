import os

folder_path = "Dataset/TESS/Processed"
file_count = sum(len(files) for _, _, files in os.walk(folder_path))
print(f"Total file di {folder_path}: {file_count}")
