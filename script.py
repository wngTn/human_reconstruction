# from PIL import Image
# import numpy as np
# import math

# # Load the OBJ file
# with open('dataset/MultiHuman/single/obj/401/401.obj', 'r') as f:
#     lines = f.readlines()

# # Collect the colors and update the OBJ lines
# colors = []
# new_lines = []
# for line in lines:
#     if line.startswith('v '):
#         parts = line.split()
#         x, y, z = map(float, parts[1:4])
#         r, g, b = map(float, parts[4:7])
#         colors.append((r, g, b))
#         new_lines.append(f'v {x} {y} {z}\n')
#     else:
#         new_lines.append(line)

# # Create the texture image
# colors = np.array(colors)
# colors = np.clip(colors, 0, 1)  # ensure colors are in the range [0, 1]
# colors = (colors * 255).astype(np.uint8)  # convert from float to 8-bit color

# # Calculate the size of the image
# num_colors = len(colors)
# side_length = math.ceil(math.sqrt(num_colors))
# colors = np.concatenate([colors, np.zeros((side_length * side_length - num_colors, 3), dtype=np.uint8)])  # pad with black
# colors = colors.reshape(side_length, side_length, 3)

# image = Image.fromarray(colors, 'RGB')

# # Save the texture image
# image.save('texture.jpg')

# # Add texture coordinates to the OBJ file
# grid_step = 1 / (side_length - 1)
# for i in range(num_colors):
#     x = (i % side_length) * grid_step
#     y = (i // side_length) * grid_step
#     new_lines.insert(i + 1, f'vt {x} {y}\n')

# # Add texture coordinate indices to the faces
# for i, line in enumerate(new_lines):
#     if line.startswith('f'):
#         parts = line.split()
#         for j in range(1, len(parts)):
#             parts[j] += f'/{j-1}'  # assuming one face per vertex, so index is just j-1
#         new_lines[i] = ' '.join(parts) + '\n'

# # Save the new OBJ file
# with open('output.obj', 'w') as f:
#     f.writelines(new_lines)


### export texture if available in .obj ###

import trimesh
import os

# obj_file = "F:/SS23/AT3DCV/MHDataset/MultiHumanDataset/multihuman_single_raw/multihuman_single/DATA401/NORMAL.obj"  # Replace with the path to your .obj file

obj_file = "F:/SS23/AT3DCV/at3dcv_project/dataset/MultiHuman/single/obj/401/401.obj"
mesh = trimesh.load_mesh(obj_file)
# print(mesh.visual.kind)
if mesh.visual.kind == 'texture' and mesh.visual.material is not None:
    texture_image = mesh.visual.material.image
    output_file = "F:/SS23/AT3DCV/MHDataset/MultiHumanDataset/multihuman_single_raw/multihuman_single/DATA401/trial_tex.png"  # Replace with the desired output file path
    texture_image.save(output_file)
    print("Texture exported successfully.")
else:
    print("The mesh does not have a texture or material.")