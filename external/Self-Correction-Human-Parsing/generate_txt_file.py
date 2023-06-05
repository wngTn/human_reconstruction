import os

def generate_image_list_txt(folder_path, txt_file_name):
    # Get the parent directory path
    parent_dir = os.path.dirname(folder_path)
    
    # Get the names of all files in the folder
    file_names = os.listdir(folder_path)
    
    # Filter the file names to include only image files
    image_names = [name for name in file_names if name.lower().endswith('.jpg')]
    # if len(image_names) == 0:
        # image_names = [name for name in file_names if name.lower().endswith('.png')]

    
    # Create the path for the txt file
    txt_file_path = os.path.join(parent_dir, txt_file_name)
    
    # Write the image names (without the .jpg extension) to the txt file
    with open(txt_file_path, 'w') as txt_file:
        for name in image_names:
            name_without_extension = os.path.splitext(name)[0]
            txt_file.write(name_without_extension + '\n')

# generate_image_list_txt('F:/SS23/AT3DCV/at3dcv_project/humanParsing/Self-Correction-Human-Parsing/mhp_extension/data/Synthetic/first_trial/global_pic', 'global_pic.txt')