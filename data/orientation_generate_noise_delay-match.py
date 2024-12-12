import numpy as np
import cv2
import random
import os

def create_delay_match(stripe_indices, output_dir, num_samples):
    # Create a blank grayscale image of 256x256 pixels with a gray background (value 127)
    image_size = 64
    background_gray_value = 127
    image = np.full((image_size, image_size), background_gray_value, dtype=np.uint8)
    cue_image = np.full((image_size, image_size), background_gray_value, dtype=np.uint8)

    # Parameters for squares
    square_size = 10
    center_distance = 20  # Distance from image center to each square center
    angles = np.arange(0, 180, 15)  # Angles at which squares will be positioned
    position_angels = np.arange(0, 360, 45)

    # Calculate the image center
    center_x, center_y = image_size // 2, image_size // 2
    frequency = 0.3  # Frequency of the sinusoidal wave

    
    square_idx = random.choice(stripe_indices)
    angle_list = []
    angle_list.sort()

    for i in range(len(stripe_indices)):
        angle_list.append(random.choice(angles))

    for seq_num in range(1, num_samples + 1):
    # Draw the squares
        for idx, square_angle in enumerate(position_angels):
            #angle_memory = []
            # Calculate the square center position
            radian = np.deg2rad(square_angle)
            square_center_x = int(center_x + center_distance * np.cos(radian))
            square_center_y = int(center_y + center_distance * np.sin(radian))
            
            # Calculate the top-left and bottom-right corners of the square
            top_left_x = square_center_x - square_size // 2
            top_left_y = square_center_y - square_size // 2
            bottom_right_x = square_center_x + square_size // 2
            bottom_right_y = square_center_y + square_size // 2

            # Generate sinusoidal stripe pattern only for the selected squares
            if idx in stripe_indices:  # Apply pattern to the randomly selected squares
                # Create sinusoidal pattern using the provided angle
                x = np.linspace(0, square_size, square_size)
                y = np.linspace(0, square_size, square_size)
                xv, yv = np.meshgrid(x, y)

                # Adjust the sinusoidal pattern calculation based on the square's angle
                angle_idx = stripe_indices.index(idx)
                angle = angle_list[angle_idx]
                #angle_memory.append(angle)
                sinusoidal_pattern = (127.5 * (1 + 
                    np.sin(2 * np.pi * frequency * (xv * np.cos(np.deg2rad(angle)) + yv * np.sin(np.deg2rad(angle)))))).astype(np.uint8)
                
                # Add transparent mosaic noise
                block_size = 2  # Size of mosaic blocks
                noise_intensity = 50  # Maximum noise intensity deviation
                alpha =  0.5 #0.6  # Blending factor (0.0 = no noise, 1.0 = full noise)

                noisy_pattern = sinusoidal_pattern.copy()

                # Number of noise blocks to apply
                num_noise_blocks = 50  # Adjust based on how much random noise you want

                for _ in range(num_noise_blocks):
                    # Randomly select the top-left corner of a block
                    y = random.randint(0, noisy_pattern.shape[0] - block_size)
                    x = random.randint(0, noisy_pattern.shape[1] - block_size)
                    
                    # Define block boundaries
                    y_end = min(y + block_size, noisy_pattern.shape[0])
                    x_end = min(x + block_size, noisy_pattern.shape[1])
                    
                    # Random noise offset
                    noise = random.randint(-noise_intensity, noise_intensity)
                    
                    # Create a block with random noise
                    noise_block = noisy_pattern[y:y_end, x:x_end] + noise
                    noise_block = np.clip(noise_block, 0, 255)
                    
                    # Blend noise block with the original block
                    noisy_pattern[y:y_end, x:x_end] = (
                        alpha * noise_block + (1 - alpha) * noisy_pattern[y:y_end, x:x_end]
                    ).astype(np.uint8)


                # Place the noisy pattern in the image
                image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = noisy_pattern


                # Place the pattern in the image
                #image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = sinusoidal_pattern

                if idx == square_idx:
                    cue_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

            else:
                # Draw a regular white square on the image
                cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)
            
        # Create a sample image
        output_path = os.path.join(output_dir, f"1_sample_{seq_num}.png")
        cv2.imwrite(output_path, image)

        cue_idx = stripe_indices.index(square_idx)
        print(stripe_indices)
        cue_angle = angle_list[cue_idx]
    
        output_path = os.path.join(output_dir, f"3_cue_{seq_num}_{cue_angle}.png")
        cv2.imwrite(output_path, cue_image)


def create_unchanged(stripe_indices, output_dir, num_samples):
    image_size = 64
    background_gray_value = 127
    frequency = 0.3
    square_size = 10
    center_distance = 20
    angles = np.arange(0, 360, 45)
    center_x, center_y = image_size // 2, image_size // 2
    angle_list = []
    angle_list.sort()

    for i in range(len(stripe_indices)):
        angle_list.append(random.choice(angles))

    for seq_num in range(1, num_samples + 1):
        image = np.full((image_size, image_size), background_gray_value, dtype=np.uint8)
        image2 = np.full((image_size, image_size), background_gray_value, dtype=np.uint8)

        for idx, square_angle in enumerate(angles):
            radian = np.deg2rad(square_angle)
            square_center_x = int(center_x + center_distance * np.cos(radian))
            square_center_y = int(center_y + center_distance * np.sin(radian))

            top_left_x = square_center_x - square_size // 2
            top_left_y = square_center_y - square_size // 2
            bottom_right_x = square_center_x + square_size // 2
            bottom_right_y = square_center_y + square_size // 2

            if idx in stripe_indices:
                x = np.linspace(0, square_size, square_size)
                y = np.linspace(0, square_size, square_size)
                xv, yv = np.meshgrid(x, y)

                angle_idx = stripe_indices.index(idx)
                angle = angle_list[angle_idx]
                sinusoidal_pattern = (127.5 * (1 + np.sin(2 * np.pi * frequency * 
                                      (xv * np.cos(np.deg2rad(angle)) + yv * np.sin(np.deg2rad(angle)))))).astype(np.uint8)
                
                
                # Add mosaic noise
                block_size = 2
                noise_intensity = 50
                alpha = 0.5
                noisy_pattern = sinusoidal_pattern.copy()

                for _ in range(50):  # Number of noise blocks
                    y_noise = random.randint(0, noisy_pattern.shape[0] - block_size)
                    x_noise = random.randint(0, noisy_pattern.shape[1] - block_size)
                    y_end = min(y_noise + block_size, noisy_pattern.shape[0])
                    x_end = min(x_noise + block_size, noisy_pattern.shape[1])
                    noise = random.randint(-noise_intensity, noise_intensity)
                    noise_block = noisy_pattern[y_noise:y_end, x_noise:x_end] + noise
                    noise_block = np.clip(noise_block, 0, 255)
                    noisy_pattern[y_noise:y_end, x_noise:x_end] = (
                        alpha * noise_block + (1 - alpha) * noisy_pattern[y_noise:y_end, x_noise:x_end]
                    ).astype(np.uint8)

                image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = noisy_pattern

                noisy_pattern = sinusoidal_pattern.copy()

                for _ in range(50):  # Number of noise blocks
                    y_noise = random.randint(0, noisy_pattern.shape[0] - block_size)
                    x_noise = random.randint(0, noisy_pattern.shape[1] - block_size)
                    y_end = min(y_noise + block_size, noisy_pattern.shape[0])
                    x_end = min(x_noise + block_size, noisy_pattern.shape[1])
                    noise = random.randint(-noise_intensity, noise_intensity)
                    noise_block = noisy_pattern[y_noise:y_end, x_noise:x_end] + noise
                    noise_block = np.clip(noise_block, 0, 255)
                    noisy_pattern[y_noise:y_end, x_noise:x_end] = (
                        alpha * noise_block + (1 - alpha) * noisy_pattern[y_noise:y_end, x_noise:x_end]
                    ).astype(np.uint8)

                image2[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = noisy_pattern
            else:
                cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)
                cv2.rectangle(image2, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)

        sample_path = os.path.join(output_dir, f"1_sample_{seq_num}.png")
        cv2.imwrite(sample_path, image)

        test_path = os.path.join(output_dir, f"3_test_{seq_num}.png")
        cv2.imwrite(test_path, image2)


def create_changed(stripe_indices, output_dir, num_samples):
    image_size = 64
    background_gray_value = 127
    frequency = 0.3
    square_size = 10
    center_distance = 20
    angles = np.arange(0, 360, 45)
    center_x, center_y = image_size // 2, image_size // 2
    angle_record = {}

    angle_list = []
    angle_list.sort()

    for i in range(len(stripe_indices)):
        angle_list.append(random.choice(angles))

    position_idx = random.choice(stripe_indices)
    changed_angle_idx = stripe_indices.index(position_idx)
    changed_angle = angle_list[changed_angle_idx]
    # Randomly select a new angle different from the original for the test square
    new_angle = random.choice([a for a in angles if a != changed_angle and a % 180 != changed_angle % 180])

    location_angle = angles[position_idx]

    for seq_num in range(1, num_samples + 1):
        image = np.full((image_size, image_size), background_gray_value, dtype=np.uint8)

        for idx, square_angle in enumerate(angles):
            radian = np.deg2rad(square_angle)
            square_center_x = int(center_x + center_distance * np.cos(radian))
            square_center_y = int(center_y + center_distance * np.sin(radian))

            top_left_x = square_center_x - square_size // 2
            top_left_y = square_center_y - square_size // 2
            bottom_right_x = square_center_x + square_size // 2
            bottom_right_y = square_center_y + square_size // 2

            if idx in stripe_indices:
                x = np.linspace(0, square_size, square_size)
                y = np.linspace(0, square_size, square_size)
                xv, yv = np.meshgrid(x, y)

                angle_idx = stripe_indices.index(idx)
                angle = angle_list[angle_idx]
                angle_record[idx] = angle
                sinusoidal_pattern = (127.5 * (1 + np.sin(2 * np.pi * frequency * 
                                      (xv * np.cos(np.deg2rad(angle)) + yv * np.sin(np.deg2rad(angle)))))).astype(np.uint8)

                # Add mosaic noise
                block_size = 2
                noise_intensity = 50
                alpha = 0.5
                noisy_pattern = sinusoidal_pattern.copy()

                for _ in range(50):  # Number of noise blocks
                    y_noise = random.randint(0, noisy_pattern.shape[0] - block_size)
                    x_noise = random.randint(0, noisy_pattern.shape[1] - block_size)
                    y_end = min(y_noise + block_size, noisy_pattern.shape[0])
                    x_end = min(x_noise + block_size, noisy_pattern.shape[1])
                    noise = random.randint(-noise_intensity, noise_intensity)
                    noise_block = noisy_pattern[y_noise:y_end, x_noise:x_end] + noise
                    noise_block = np.clip(noise_block, 0, 255)
                    noisy_pattern[y_noise:y_end, x_noise:x_end] = (
                        alpha * noise_block + (1 - alpha) * noisy_pattern[y_noise:y_end, x_noise:x_end]
                    ).astype(np.uint8)

                image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = noisy_pattern
            else:
                cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)
            
        test_image = image.copy()

        

        # Calculate the square center position for the test
        radian = np.deg2rad(location_angle)
        square_center_x = int(center_x + center_distance * np.cos(radian))
        square_center_y = int(center_y + center_distance * np.sin(radian))

        # Calculate the top-left and bottom-right corners of the square
        top_left_x = square_center_x - square_size // 2
        top_left_y = square_center_y - square_size // 2
        bottom_right_x = square_center_x + square_size // 2
        bottom_right_y = square_center_y + square_size // 2

        # Create a new sinusoidal pattern for the test image with the new angle
        x = np.linspace(0, square_size, square_size)
        y = np.linspace(0, square_size, square_size)
        xv, yv = np.meshgrid(x, y)

        # Adjust the sinusoidal pattern calculation based on the new angle
        sinusoidal_pattern = (127.5 * (1 + 
            np.sin(2 * np.pi * frequency * (xv * np.cos(np.deg2rad(new_angle)) + yv * np.sin(np.deg2rad(new_angle)))))).astype(np.uint8)

        # Add mosaic noise
        block_size = 2
        noise_intensity = 50
        alpha = 0.5
        noisy_pattern = sinusoidal_pattern.copy()

        for _ in range(50):  # Number of noise blocks
            y_noise = random.randint(0, noisy_pattern.shape[0] - block_size)
            x_noise = random.randint(0, noisy_pattern.shape[1] - block_size)
            y_end = min(y_noise + block_size, noisy_pattern.shape[0])
            x_end = min(x_noise + block_size, noisy_pattern.shape[1])
            noise = random.randint(-noise_intensity, noise_intensity)
            noise_block = noisy_pattern[y_noise:y_end, x_noise:x_end] + noise
            noise_block = np.clip(noise_block, 0, 255)
            noisy_pattern[y_noise:y_end, x_noise:x_end] = (
                alpha * noise_block + (1 - alpha) * noisy_pattern[y_noise:y_end, x_noise:x_end]
            ).astype(np.uint8)

        test_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = noisy_pattern

        sample_path = os.path.join(output_dir, f"1_sample_{seq_num}.png")
        cv2.imwrite(sample_path, image)

        test_path = os.path.join(output_dir, f"3_test_{seq_num}.png")
        cv2.imwrite(test_path, test_image)


def create_same_dataset(num_samples, n, output_dir, change):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Randomly select an angle for all samples in this dataset  
    stripe_indices = random.sample(range(8), n)  # Randomly choose n unique indices
    stripe_indices.sort()
    if not change:
        create_unchanged(stripe_indices, output_dir, num_samples)
    
    elif change:
        create_changed(stripe_indices, output_dir, num_samples)
        
    # Create a delay image (gray background)
    for seq_num in range(1, num_samples + 1):
        delay_path = os.path.join(output_dir, f"2_delay_{seq_num}.png")
        gray_image = np.full((64, 64), 127, dtype=np.uint8)
        cv2.imwrite(delay_path, gray_image)

    #create_delay_match(stripe_indices, output_dir, num_samples)
        
        
    # Randomly select indices for squares that will have stripes
    # create_image_with_n_striped_squares(stripe_indices, output_dir, num_samples, test_or_sample="test")

def generate_data(num_sequences=10, output_dir_base="output_images"):
    # Generate data for different numbers of striped squares (1, 2, 4, 8)
    change = False
    for n in [1, 2, 4, 8]:
        for folder_num in range(1, 1001):  # Create 30 folders
            output_dir = os.path.join(output_dir_base, str(n), f"{folder_num}")
            create_same_dataset(num_sequences, n, output_dir, change)
            print(f"Generated images for {n} striped squares in folder {folder_num}.")

# Run the data generation with 10 sequences
generate_data(num_sequences=10, output_dir_base='./cap-Yudi/data/orientation/unchanged')
