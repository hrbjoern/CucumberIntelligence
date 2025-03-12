import os
import shutil 
import glob
from random import shuffle
from math import floor, ceil

# subfunctions: 
def create_images_and_labels_folders(source_folder):
    """
    _summary_ Function to create "images" and "labels" subfolders in the source folder.

    Args:
        source_folder: train, val or test
    """
    # Define the target subfolder path
    target_image_folder = os.path.join(source_folder, 'images')
    target_label_folder = os.path.join(source_folder, 'labels')

    # Create the "images" and "labels" subfolders if they don't exist
    os.makedirs(target_image_folder, exist_ok=False)
    os.makedirs(target_label_folder, exist_ok=False)
    print(f"Created {target_image_folder} and {target_label_folder}.")

def get_train_val_test_lists(file_list, split=(0.7,0.9)):
    #split = 0.7
    lower_split_index = floor(len(file_list) * split[0])
    upper_split_index = floor(len(file_list) * split[1])
    train = file_list[:lower_split_index]
    val = file_list[lower_split_index:upper_split_index]
    test = file_list[upper_split_index:]
    return train, val, test

def move_images_and_labels_to_folders(imagelist, destination_folder, labeltype):
    """
    _summary_ Function to move image and label files to the corresponding subfolders.
    """

    if labeltype == 'boxes':
        label_folder = 'converted_boxes'
    elif labeltype == 'segs':
        label_folder = 'converted_segmentations'
    else:
        print("Error: Invalid label type. Please choose 'boxes' or 'segs'.")
        return 

    # Define the target subfolder paths:
    target_image_folder = os.path.join(destination_folder, 'images')
    target_label_folder = os.path.join(destination_folder, 'labels')

    for image_name in imagelist:
        # Full path to the file
        #image_file_path = os.path.join(source_folder, image_filename)
        image_file_path = image_name + '.png'
        label_file_path = os.path.join(label_folder, image_name+'.txt')

        # Check if it's a file (not a folder)
        if os.path.isfile(image_file_path):
            # Move the files to the subfolders:
            shutil.move(image_file_path, target_image_folder)
            #print(f"Moved {image_file_path} to {target_image_folder}.")
            shutil.move(label_file_path, target_label_folder)
            #print(f"Moved {label_file_path} to {target_label_folder}.")
            # TODO: prints entfernen!
        else: 
            print(f"{image_file_path} is not an image. Skipping!")
            #return
    print(f"Moved {len(imagelist)} images and labels to folders.")

def prepare_image_name_lists(labeltype):
    """_summary_ Function to prepare image name lists for train, val and test folders.
    _output_ Prints the number of labels and labelled images found.
    _returns_ trainlist, vallist, testlist
    """

    if labeltype == 'boxes':
        label_folder = 'converted_boxes'
    elif labeltype == 'segs':
        label_folder = 'converted_segmentations'
    else:
        print("Error: Invalid label type. Please choose 'boxes' or 'segs'.")
        return 

    # 1) Prepare image name lists:
    current_folder = os.getcwd()
    print("Current folder is:", current_folder)

    # labelfiles = os.listdir(os.path.join(current_folder, 
    #                                      'converted_segmentation_labels'))
    labelfiles = os.listdir(label_folder)
    labelnames = [os.path.splitext(name)[0] for name in labelfiles]

    print(f"Found {len(labelfiles)} labels.")

    # Sanity check for small image sets:
    # Filter for .png and .jpg files
    image_extensions = ('.png', '.jpg')
    imagenames = [os.path.splitext(file)[0] for file in os.listdir(current_folder) 
                  if file.lower().endswith(image_extensions)]
    imagenames = [name for name in imagenames if name in labelnames]

    # Shuffle image names for splitting: 
    shuffle(imagenames) # shuffles in place!

    print(f"Found {len(imagenames)} labelled images.") #\n {imagenames}")

    #print(get_train_val_test_lists(imagenames))
    return get_train_val_test_lists(imagenames)


#
# Main Sammel-function: 
#
def prepare_yolo_data_folders(labeltype):
    """
    _summary_ Function to be called from within Data folder containing
                 images and converted_labels folder. 
    _output_ Creates a folder structure for YOLO training data.
    _returns_ True on success, False on failure.
    """
    print(f"Preparing files and folders for training with {labeltype} labels.")
    
    if labeltype not in ('boxes', 'segs'):
        print("Error: Invalid label type in prepare_yolo_data_folders(labeltype). Please choose 'boxes' or 'segs'.")
        return False

    # 1) Get the image name lists:
    trainlist, vallist, testlist = prepare_image_name_lists(labeltype)

    # 2) Prepare image and label folders for train, val and test: 
    foldernames = ['train', 'val', 'test']

    for foldername in foldernames: 
        create_images_and_labels_folders(foldername)

    # 3) Move images and labels to the subfolders:
    for foldername, imagelist in zip(foldernames, [trainlist, vallist, testlist]):
        move_images_and_labels_to_folders(imagelist, foldername, labeltype)

    # Insert assertions, exception handling etc. here. ;-) 
    print("Data handling finished.")
    return True




#####################################

if __name__ == '__main__':
    prepare_yolo_data_folders('boxes')
    #prepare_image_name_lists()
    # print("Files have been moved.")
    # print("Image name lists have been prepared.")
    # print("Done.")
    pass

