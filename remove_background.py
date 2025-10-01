import os
from rembg import remove
from PIL import Image
import numpy as np
import cv2

def remove_background_nested(input_dir, output_base_dir='background_removed'):
    # Ensure the base directory for output exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Walk through all directories and subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir):
        # Construct the path for the new directory
        rel_path = os.path.relpath(root, input_dir)  # Get relative path to maintain directory structure
        output_dir = os.path.join(output_base_dir, rel_path)

        # Prepare the output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Process each file in the current directory
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                
                # Open the image
                input_image = Image.open(input_path)
                
                # Remove the background from the image
                output_image = remove(input_image)
                
                # Convert to array and save as image
                output_array = np.asarray(output_image)
                cv2.imwrite(output_path, output_array)

# Usage example:
remove_background_nested('copied images')
