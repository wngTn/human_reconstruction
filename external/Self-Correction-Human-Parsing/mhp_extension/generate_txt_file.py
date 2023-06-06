import os
import argparse

def generate_image_list_txt(folder_path, txt_file_name):
    # Get the parent directory path
    parent_dir = os.path.dirname(folder_path)
    
    # Get the names of all files in the folder
    file_names = os.listdir(folder_path)
    
    # Filter the file names to include only image files
    image_names = [name for name in file_names if name.lower().endswith('.jpg') or name.lower().endswith('.png')]
    print(f"Found {len(image_names)} images.")

    
    # Create the path for the txt file
    txt_file_path = os.path.join(parent_dir, txt_file_name)
    
    # Write the image names (without the .jpg extension) to the txt file
    with open(txt_file_path, 'w') as txt_file:
        for name in image_names:
            name_without_extension = os.path.splitext(name)[0]
            txt_file.write(name_without_extension + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a txt file with the list of image names in a directory.')
    parser.add_argument('--folder_path', required=True, type=str, help='The path to the image directory.')
    parser.add_argument('--txt_file_name', required=True, type=str, help='The name of the txt file to create.')

    args = parser.parse_args()

    generate_image_list_txt(args.folder_path, args.txt_file_name)