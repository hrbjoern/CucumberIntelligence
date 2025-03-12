import os
import shutil
import glob


# TODO: if needed, build train_test_split functionality here.

#conv_label_folder = 'converted_labels'
conv_label_folder = 'converted_segmentation_labels'


# Function to move files to the subfolders.
def move_images_and_labels(source_folder):
    # Define the target subfolder path
    target_image_folder = os.path.join(source_folder, 'images')
    target_label_folder = os.path.join(source_folder, 'labels')

    # Create the "images" and "labels" subfolders if they don't exist
    os.makedirs(target_image_folder, exist_ok=True)
    os.makedirs(target_label_folder, exist_ok=True)

    # List all files in the source folder
    for image_filename in os.listdir(source_folder):
        # Full path to the file
        image_file_path = os.path.join(source_folder, image_filename)

        # Check if it's a file (not a folder)
        if os.path.isfile(image_file_path):

            # Assert that the corresponding label file exists:
            label_file_path = os.path.join(conv_label_folder, image_filename.replace('.png', '.txt'))
            assert os.path.isfile(label_file_path), f"Label file {label_file_path} not found."

            # Move the files to the subfolders:
            shutil.move(image_file_path, os.path.join(target_image_folder, image_filename))
            shutil.move(label_file_path, target_label_folder)
        else: 
            print(f"{image_file_path} is not an image. Skipping!")
            #return

    # Check for number of files in the subfolders: 
    num_images = len(os.listdir(target_image_folder))
    num_labels = len(os.listdir(target_label_folder))
    if num_images != num_labels:
        print(f"Number of images ({num_images}) and labels ({num_labels}) do not match.")
    else:
        print(f"Moved {num_images} images and labels to {source_folder}.")

# Move files in both "train" and "val" folders
move_images_and_labels('train')
move_images_and_labels('val')
move_images_and_labels('test')


# print("Files have been moved.")
