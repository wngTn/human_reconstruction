import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def depth_to_npz(folder_path, target_path):
    """
    Convert depth images to numpy arrays and reorganize in a target folder
    """
    os.makedirs(target_path, exist_ok=True)

    for dm in tqdm(os.listdir(folder_path)):
        if dm.endswith('.png'):
            depth_map = Image.open(os.path.join(folder_path, dm))
            depth_map_array = np.array(depth_map)
            split_name = dm.split('_')
            save_dir = split_name[1]
            save_name = split_name[2]
            os.makedirs(os.path.join(target_path, save_dir), exist_ok=True)
            np.savez(os.path.join(target_path, save_dir, f'{save_name}.npz'), depth_map_array=depth_map_array)
    print('Depth folder transformed and reorganized.')


def img_reorganize(folder_path, target_path):
    """
    Reorganize image files in a target folder
    """
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
    """
    Reorganize normal files in a target folder
    """
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
    """
    Reorganize mask files in a target folder
    """
    os.makedirs(target_path, exist_ok=True)

    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.png'):
            split_name = file_name.split('_')
            save_dir = split_name[-2]
            save_name = split_name[-1]
            os.makedirs(os.path.join(target_path, save_dir), exist_ok=True)
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(target_path, save_dir, f'{save_name}')
            shutil.copy2(old_file_path, new_file_path)
    print('Mask files reorganized.')

def copy_image(source_folder, destination_folder):
    # Ensures that the destination directory exists, if it doesn't, creates it.
    os.makedirs(destination_folder, exist_ok=True)

    count = 0
    # Walks through all files in the source directory
    for root, dirs, files in tqdm(os.walk(source_folder), desc="Copying Images"):
        for file in files:
            # Only process files that ends with .jpg
            if file.endswith('.png'):
                # Define the source file path
                source_path = os.path.join(root, file)
                # Extract subfolder name to be used in new file name
                subfolder_name = os.path.basename(root)
                # Create new filename
                new_filename = f'{subfolder_name}_{file}'
                # Define the destination file path
                destination_path = os.path.join(destination_folder, new_filename)
                # Copy the file from source to destination
                shutil.copyfile(source_path, destination_path)
                
                # Count the copied files
                count += 1
                # Limit the number of files copied to 24
                if count == 24:
                    break
        if count == 24:
            break

    print(f"Copied {count} files from {source_folder} to {destination_folder}.")


def main(args):
    if args.depth_folder and args.depth_target:
        depth_to_npz(args.depth_folder, args.depth_target)
    if args.img_folder and args.img_target:
        img_reorganize(args.img_folder, args.img_target)
    if args.normal_folder and args.normal_target:   
        normal_reorganize(args.normal_folder, args.normal_target)
    if args.mask_folder and args.mask_target:
        mask_reorganize(args.mask_folder, args.mask_target)
    if args.source_image and args.destination_image:
        copy_image(args.source_image, args.destination_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reorganize and convert files')
    parser.add_argument('--depth_folder', type=str, default=None, help='Source folder for depth images')
    parser.add_argument('--depth_target', type=str, default=None, help='Target folder for depth npz files')
    parser.add_argument('--img_folder', type=str, default=None, help='Source folder for images')
    parser.add_argument('--img_target', type=str, default=None, help='Target folder for images')
    parser.add_argument('--normal_folder', type=str, default=None, help='Source folder for normal files')
    parser.add_argument('--normal_target', type=str, default=None, help='Target folder for normal files')
    parser.add_argument('--mask_folder', type=str, default=None, help='Source folder for mask files')
    parser.add_argument('--mask_target', type=str, default=None, help='Target folder for mask files')
    parser.add_argument("-src_img", "--source_image", help="Source directory", default=None)
    parser.add_argument("-dst_img", "--destination_image", help="Destination directory", default=None)
    args = parser.parse_args()
    main(args)


