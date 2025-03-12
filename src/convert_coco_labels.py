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

# Function to normalize segmentation points to YOLO format
def normalize_segmentation(segmentation, img_width, img_height):
    normalized_segmentation = []
    for i in range(0, len(segmentation), 2):
        x = segmentation[i] / img_width
        y = segmentation[i + 1] / img_height
        normalized_segmentation.append(f"{x} {y}")
    return normalized_segmentation

# Directory to save YOLO segmentation annotations
box_dir = 'converted_boxes'
seg_dir = 'converted_segmentations'
os.makedirs(box_dir, exist_ok=False)
os.makedirs(seg_dir, exist_ok=False)

# Iterate over images and create image_id_to_filename dictionary: 
image_file_id = {}
for image in coco_data['images']:
    image_id = image['id']
    filename = image['file_name']
    image_file_id[image_id] = os.path.splitext(filename)[0]

print("image_file_id list:")
print(image_file_id)

# Iterate over annotations and convert
for annotation in coco_data['annotations']:
    img_id = annotation['image_id']
    image_info = next(img for img in coco_data['images'] 
                      if img['id'] == img_id)
    img_width, img_height = image_info['width'], image_info['height']
    
    # Convert bounding box
    yolo_bbox = coco_to_yolo_bbox(annotation['bbox'], 
                                  img_width, 
                                  img_height)
    
    # YOLO format: <category_id> <x_center> <y_center> <width> <height> (bbox)
    #          OR: <category_id> <x1> <y1> ... <xn> <yn> (segmentations)
    box_line = f"{annotation['category_id']} " + " ".join(map(str, yolo_bbox)) + "\n"
            
    # Save to a file corresponding to the image ID
    box_file = os.path.join(box_dir, f"{image_file_id[img_id]}.txt")
    with open(box_file, 'a') as f:
        f.write(box_line)

    # Normalize segmentation points
    if annotation['segmentation']:
        segmentation = annotation['segmentation'][0]  # Assume it's polygon (not RLE)
        yolo_segmentation = normalize_segmentation(segmentation, img_width, img_height)
    
        seg_line = f"{annotation['category_id']} " + " ".join(yolo_segmentation) + "\n"

        # Save to a file corresponding to the image ID
        seg_file = os.path.join(seg_dir, f"{image_file_id[img_id]}.txt")
        with open(seg_file, 'a') as f:
            f.write(seg_line)
    else:
        print(f"Skipping annotation {annotation['id']} of file {label_file} with no segmentation!") 
