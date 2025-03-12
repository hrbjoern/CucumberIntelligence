import json
import os

# Load COCO annotations
with open('annotations.json', 'r') as f:
    coco_data = json.load(f)

# Function to convert COCO bbox to YOLO format
def coco_to_yolo_bbox(bbox, img_width, img_height):
    x_min, y_min, box_width, box_height = bbox
    x_center = x_min + box_width / 2
    y_center = y_min + box_height / 2
    return [
        x_center / img_width,
        y_center / img_height,
        box_width / img_width,
        box_height / img_height
    ]

# Directory to save YOLO annotations
output_dir = 'converted_labels'
os.makedirs(output_dir, exist_ok=True)

# Iterate over images and create image_id_to_filename dictionary: 
image_file_id = {}
for image in coco_data['images']:
    image_id = image['id']
    filename = image['file_name']
    image_file_id[image_id] = os.path.splitext(filename)[0]

print("image_file_id dict:")
print(image_file_id)

# Iterate over annotations and convert
for annotation in coco_data['annotations']:
    img_id = annotation['image_id']
    image_info = next(img for img in coco_data['images'] 
                      if img['id'] == img_id)
    img_width, img_height = image_info['width'], image_info['height']
    
    yolo_bbox = coco_to_yolo_bbox(annotation['bbox'], 
                                  img_width, 
                                  img_height)
    
    # YOLO format: <category_id> <x_center> <y_center> <width> <height>
    yolo_line = f"{annotation['category_id']} " + " ".join(map(str, yolo_bbox)) + "\n"
    
    # Save to a file corresponding to the image ID
    label_file = os.path.join(output_dir, 
                              f"{image_file_id[img_id]}.txt")
    with open(label_file, 'a') as f:
        f.write(yolo_line)
