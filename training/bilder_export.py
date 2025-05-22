import os
import shutil


source_root = "acquisitions"

target_folder = "extracted_pngs"


os.makedirs(target_folder, exist_ok=True)

for s in range(1, 5): 
    for g in range(1, 12):  
        subfolder = os.path.join(source_root, f"S{s}", f"G{g}")
        if os.path.isdir(subfolder):
            for file in os.listdir(subfolder):
                if file.lower().endswith(".png"):
                    full_path = os.path.join(subfolder, file)
                    target_path = os.path.join(target_folder, f"S{s}_G{g}_{file}")
                    shutil.copy(full_path, target_path)
