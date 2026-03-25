import cv2

def center_crop(image_path, output_path, crop_width, crop_height):
    # 1. Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Get the original dimensions
    # img.shape returns (height, width, channels)
    original_height, original_width = img.shape[:2]

    # 3. Check if the requested crop is larger than the image itself
    if crop_width > original_width or crop_height > original_height:
        print("Error: Crop size is larger than original image dimensions.")
        return

    # 4. Calculate the center coordinates
    center_x, center_y = original_width // 2, original_height // 2
    
    # 5. Calculate the starting X and Y for the slice
    start_x = 3202
    start_y = 1402
    
    # 6. Slice the NumPy array to perform the crop
    # Syntax: array[startY:endY, startX:endX]
    cropped_img = img[start_y : start_y + crop_height, start_x : start_x + crop_width]

    # 7. Save and display the result
    cv2.imwrite(output_path, cropped_img)
    print(f"Successfully saved center crop to {output_path}")

center_crop(
    image_path="noisy_3_guided.png", 
    output_path="noisy_3_guided_c.png",
    crop_width=355,
    crop_height=330
)
