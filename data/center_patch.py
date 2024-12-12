import os
import cv2
import numpy as np
import random

def create_delay_match(output_dir, num_samples):
    os.makedirs(output_dir, exist_ok=True)
    # Create a blank grayscale image of 64x64 pixels with a gray background (value 127)
    image_size = 64
    background_gray_value = 127
    square_size = 50  # Size of the central square
    frequency = 0.3  # Frequency of the sinusoidal wave
    noise_intensity = 50  # Maximum noise intensity deviation
    alpha = 0.5  # Blending factor for mosaic noise
    angle = random.choice(np.arange(0, 180, 5))
    
    for seq_num in range(1, num_samples + 1):
        # Initialize the main image and cue image
        image = np.full((image_size, image_size), background_gray_value, dtype=np.uint8)
        cue_image = np.full((image_size, image_size), background_gray_value, dtype=np.uint8)

        # Define the central square's position
        center_x, center_y = image_size // 2, image_size // 2
        top_left_x = center_x - square_size // 2
        top_left_y = center_y - square_size // 2
        bottom_right_x = center_x + square_size // 2
        bottom_right_y = center_y + square_size // 2

        # Generate sinusoidal stripe pattern
        x = np.linspace(0, square_size, square_size)
        y = np.linspace(0, square_size, square_size)
        xv, yv = np.meshgrid(x, y)
          # Random angle for the stripe pattern

        sinusoidal_pattern = (127.5 * (1 + 
            np.sin(2 * np.pi * frequency * (xv * np.cos(np.deg2rad(angle)) + yv * np.sin(np.deg2rad(angle)))))).astype(np.uint8)

        # Add transparent mosaic noise to the sinusoidal pattern
        noisy_pattern = sinusoidal_pattern.copy()
        block_size = 2  # Size of mosaic blocks
        num_noise_blocks = 50  # Number of noise blocks to apply

        for _ in range(num_noise_blocks):
            # Randomly select the top-left corner of a block
            noise_y = random.randint(0, noisy_pattern.shape[0] - block_size)
            noise_x = random.randint(0, noisy_pattern.shape[1] - block_size)
            
            # Define block boundaries
            noise_y_end = min(noise_y + block_size, noisy_pattern.shape[0])
            noise_x_end = min(noise_x + block_size, noisy_pattern.shape[1])
            
            # Random noise offset
            noise = random.randint(-noise_intensity, noise_intensity)
            
            # Create a block with random noise
            noise_block = noisy_pattern[noise_y:noise_y_end, noise_x:noise_x_end] + noise
            noise_block = np.clip(noise_block, 0, 255)
            
            # Blend noise block with the original block
            noisy_pattern[noise_y:noise_y_end, noise_x:noise_x_end] = (
                alpha * noise_block + (1 - alpha) * noisy_pattern[noise_y:noise_y_end, noise_x:noise_x_end]
            ).astype(np.uint8)

        # Place the noisy pattern in the central square of the image
        image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = noisy_pattern

        # Create the cue image by marking the central square
        cue_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

        # Save the generated images
        output_path_sample = os.path.join(output_dir, f"1_sample_{seq_num}_{angle}.png")
        print(output_path_sample)
        cv2.imwrite(output_path_sample, image)

        output_path_cue = os.path.join(output_dir, f"3_cue_{seq_num}_{angle}.png")
        # cv2.imwrite(output_path_cue, cue_image)

        print(f"Generated sample and cue images for sequence {seq_num} with angle {angle}.")

def generate_data(num_sequences=10, output_dir_base="output_images"):
    for folder_num in range(1, 1001):  # Create 30 folders
        output_dir = os.path.join(output_dir_base, f"{folder_num}")
        create_delay_match(output_dir, num_sequences)
        print(f"Generated images for striped squares in folder {folder_num}.")
        # for seq_num in range(1, 11):
        #     delay_path = os.path.join(output_dir, f"2_delay_{seq_num}.png")
        #     gray_image = np.full((64, 64), 127, dtype=np.uint8)
        #     cv2.imwrite(delay_path, gray_image)

generate_data(num_sequences=1, output_dir_base='./cap-Yudi/data/center/1')



