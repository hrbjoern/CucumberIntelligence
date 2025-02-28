import numpy as np

def parse_yolo_file(file_path):
    bounding_boxes = []
    segmentations = []
    
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            class_id = int(values[0])
            
            if len(values) == 5:
                # Bounding box
                x_center, y_center, width, height = values[1:]
                bounding_boxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
            elif len(values) > 5:
                # Segmentation
                points = np.array(values[1:]).reshape(-1, 2)
                segmentations.append({
                    'class_id': class_id,
                    'points': points
                })
            else:
                print(f"Warning: Invalid line format: {line}")
    
    return bounding_boxes, segmentations

def print_results(bounding_boxes, segmentations):
    print("Bounding Boxes:")
    for idx, box in enumerate(bounding_boxes):
        print(f"  Box {idx + 1}: Class {box['class_id']}, "
              f"Center ({box['x_center']:.2f}, {box['y_center']:.2f}), "
              f"Size ({box['width']:.2f}, {box['height']:.2f})")
    
    print("\nSegmentations:")
    for idx, seg in enumerate(segmentations):
        print(f"  Segmentation {idx + 1}: Class {seg['class_id']}, "
              f"Points: {seg['points'].shape[0]}")
        print(f"    First point: ({seg['points'][0][0]:.2f}, {seg['points'][0][1]:.2f})")

# Example usage
file_path = 'path/to/your/yolo_labels.txt'
bounding_boxes, segmentations = parse_yolo_file(file_path)
print_results(bounding_boxes, segmentations)
