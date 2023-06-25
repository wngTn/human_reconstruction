import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

def depth_to_npz(folder_path, target_path):
    folder_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/Depth"
    folder_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/depth"
    os.makedirs(target_path, exist_ok=True)

    for dm in tqdm(os.listdir(folder_path)):
        if dm.endswith('.png'):
            depth_map = Image.open(os.path.join(folder_path, dm))
            depth_map_array = np.array(depth_map)
            split_name = dm.split('_')
            # save_dir = str(id)
            save_dir = split_name[1]
            save_name = split_name[2]
            os.makedirs(os.path.join(target_path, save_dir), exist_ok=True)
            np.savez(os.path.join(target_path, save_dir, f'{save_name}.npz'), depth_map_array=depth_map_array)
    print('Depth folder transformed and reorganized.')



def img_reorganize(folder_path, target_path):
    # folder_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial"
    # target_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/img"
    os.makedirs(target_path, exist_ok=True)

    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.png'):
            split_name = file_name.split('_')
            save_dir = split_name[1]
            save_name = split_name[2]
            os.makedirs(os.path.join(target_path, save_dir), exist_ok=True)
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(target_path, save_dir, f'{save_name}.png')
            shutil.copy2(old_file_path, new_file_path)
    print('Img files reorganized.')


def normal_reorganize(folder_path, target_path):
    # folder_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/Normal"
    # target_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/normal"
    os.makedirs(target_path, exist_ok=True)

    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.png'):
            split_name = file_name.split('_')
            save_dir = split_name[1]
            save_name = split_name[2]
            os.makedirs(os.path.join(target_path, save_dir), exist_ok=True)
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(target_path, save_dir, f'{save_name}.png')
            shutil.copy2(old_file_path, new_file_path)
    print('Normal files reorganized.')



def mask_reorganize(folder_path, target_path):
    os.makedirs(target_path, exist_ok=True)

    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.png'):
            split_name = file_name.split('_')
            save_dir = split_name[1]
            save_name = split_name[2]
            os.makedirs(os.path.join(target_path, save_dir), exist_ok=True)
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(target_path, save_dir, f'{save_name}')
            shutil.copy2(old_file_path, new_file_path)
    print('Mask files reorganized.')



if __name__ == '__main__':
    depth_to_npz(folder_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/Depth_original", target_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/img")
    img_reorganize(folder_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial", target_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/img")
    normal_reorganize(folder_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/Normal_original", target_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/normal")
    mask_reorganize(folder_path = "F:/SS23/AT3DCV/at3dcv_project/humanParsing/Self-Correction-Human-Parsing/mhp_extension/data/Synthetic/first_trial/mhp_fusion_parsing/global_tag", target_path = "F:/SS23/AT3DCV/at3dcv_project/data/Synthetic/first_trial/mask")






