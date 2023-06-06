import cv2
import os
import numpy as np

## crop bounding boxes 

# # Read the original image
# image_path = 'F:/SS23\AT3DCV/at3dcv_project/humanParsing/Self-Correction-Human-Parsing/mhp_extension/data/Test3Dataset_0/mhp_fusion_parsing/global_tag/379_0.png'
# image = cv2.imread(image_path)

# # Read the mask information from the text file
# txt_file_path = 'F:/SS23\AT3DCV/at3dcv_project/humanParsing/Self-Correction-Human-Parsing/mhp_extension/data/Test3Dataset_0/mhp_fusion_parsing/global_tag/379_0.txt'
# with open(txt_file_path, 'r') as file:
#     mask_info = file.readlines()

# # Create separate images for each mask
# for i, line in enumerate(mask_info):
#     # Parse the mask coordinates
#     confidence, y_min, x_min, y_max, x_max = map(float, line.split())

#     # Extract the mask region from the original image
#     mask = image[int(y_min):int(y_max), int(x_min):int(x_max)]

#     # Save the mask image
#     mask_image_path = f'mask_{i+1}.jpg'
#     cv2.imwrite(mask_image_path, mask)

#     # Display the mask image
#     cv2.imshow(f'Mask {i+1}', mask)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()





## crop on colors

# Read the original image
image_path = 'F:/SS23\AT3DCV/at3dcv_project/humanParsing/Self-Correction-Human-Parsing/mhp_extension/data/Test3Dataset_0/mhp_fusion_parsing/global_tag/1000_0.png'
image = cv2.imread(image_path)

# Define the color ranges for red, yellow, and green masks
red_lower = (0, 128, 128)
red_upper = (0, 255, 255)

yellow_lower = (30, 128, 128)
yellow_upper = (45, 255, 255)

green_lower = (60, 128, 128)
green_upper = (90, 255, 255)


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
output_folder = 'output_folder'
os.makedirs(output_folder, exist_ok=True)

cv2.imwrite(os.path.join(output_folder, 'red_masked_image.png'), red_masked_image)
cv2.imwrite(os.path.join(output_folder, 'yellow_masked_image.png'), yellow_masked_image)
cv2.imwrite(os.path.join(output_folder, 'green_masked_image.png'), green_masked_image)
