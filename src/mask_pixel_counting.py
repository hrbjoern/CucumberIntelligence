# From: https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/?h=annotator.seg_bbox#are-there-any-datasets-provided-by-ultralytics-suitable-for-training-yolov8-models-for-instance-segmentation-and-tracking


from ultralytics import YOLO

# Load your model and run inference to get results
model = YOLO('yolov8n-seg.pt')
results = model('./test.jpg')

# Check if masks are present
if results[0].masks is not None:
    # Convert masks to NumPy array correctly
    masks_numpy = results[0].masks.cpu().numpy()  # Make sure this conversion is correct

    # Assuming you now have a list of numpy arrays for each mask
    # You'll need to process each mask individually
    pixel_counts = [mask.sum() for mask in masks_numpy]  # This should work now

    print(f"Pixel counts for each mask: {pixel_counts}")
else:
    print("No masks found in the results.")


masks_numpy = results[0].masks.cpu().numpy()  # Convert masks to NumPy array
pixel_counts = masks_numpy.sum(axis=(1, 2))  # Sum the pixels in each mask
print(f"Target pixel count: {pixel_counts}")