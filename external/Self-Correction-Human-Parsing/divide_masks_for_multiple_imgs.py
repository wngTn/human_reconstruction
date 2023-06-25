import os
import cv2
import numpy as np

def extract_masked_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the color ranges for red, yellow, and green masks
    red_lower = (0, 128, 128)
    red_upper = (0, 255, 255)

    yellow_lower = (30, 128, 128)
    yellow_upper = (45, 255, 255)

    green_lower = (60, 128, 128)
    green_upper = (90, 255, 255)

    # Iterate over the images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Convert the image to HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Generate the red mask
            red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
            red_masked_image = np.zeros_like(image)
            red_masked_image[np.where(red_mask == 255)] = (255, 255, 255)

            # Generate the yellow mask
            yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
            yellow_masked_image = np.zeros_like(image)
            yellow_masked_image[np.where(yellow_mask == 255)] = (255, 255, 255)

            # Generate the green mask
            green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
            green_masked_image = np.zeros_like(image)
            green_masked_image[np.where(green_mask == 255)] = (255, 255, 255)

            # Save the masked images
            output_filename = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(output_folder, f'red_{output_filename}.png'), red_masked_image)
            cv2.imwrite(os.path.join(output_folder, f'yellow_{output_filename}.png'), yellow_masked_image)
            cv2.imwrite(os.path.join(output_folder, f'green_{output_filename}.png'), green_masked_image)

# Specify the input and output folders
input_folder = 'F:/SS23/AT3DCV/at3dcv_project/humanParsing/Self-Correction-Human-Parsing/mhp_extension/data/Test3Dataset/mhp_fusion_parsing/global_tag'
output_folder = 'F:/SS23/AT3DCV/at3dcv_project/dataset/trial/temp'

# Extract masked images
extract_masked_images(input_folder, output_folder)
