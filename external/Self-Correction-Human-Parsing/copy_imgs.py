import os
import shutil

# CHANGE SOURCE AND DESTINATION FOLDERS 
# source_folder = "F:/SS23/AT3DCV/MHDataset/MultiHumanDataset/Real-World-Capture/three/person2/img"
source_folder = "F:/SS23/AT3DCV/MHDataset/MultiHumanDataset/Real-World-Capture/zyx_single/img"
destination_folder = 'F:/SS23/AT3DCV/at3dcv_project/humanParsing/Self-Correction-Human-Parsing/mhp_extension/data/TestSingleDataset/global_pic'
os.makedirs(destination_folder, exist_ok=True)

# Create directories if they don't exist
# test_dataset_dir = os.path.join(destination_folder, 'TestSingleDataset')
# global_pic_dir = os.path.join(destination_folder, 'global_pic')

# os.makedirs(test_dataset_dir, exist_ok=True)
# os.makedirs(global_pic_dir, exist_ok=True)

count = 0
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.jpg'):
            source_path = os.path.join(root, file)
            subfolder_name = os.path.basename(root)
            new_filename = f'{subfolder_name}_{file}'
            destination_path = os.path.join(destination_folder, new_filename)
            shutil.copyfile(source_path, destination_path)
            
            count += 1
            if count == 24:
                break
    if count == 24:
        break