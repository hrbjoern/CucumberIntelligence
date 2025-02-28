import cv2 
import sys, datetime
from ultralytics import YOLO, solutions

# Load the YOLO model: 
modelpath = 'Models/2024-09-28_2145_boxes_best.onnx'
model = YOLO(modelpath, 'detect')

# Confidence threshold: 
confidence = 0.4

# Define the counting line: 
line_points = [(500, 50), (500, 400)]


# Capture video from the default webcam (usually index 0)
cap = cv2.VideoCapture(0)
# Try to set input frame rate: 
desired_fps = 4
cap.set(cv2.CAP_PROP_FPS, desired_fps)

desired_px = (640, 480)


# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Video writer, output file: 
from datetime import date
datestr = date.today().strftime("%Y-%m-%d")+"_"
outfilename = "FinalResults/"+datestr+"webcam_counting_output.avi"


video_writer = cv2.VideoWriter(outfilename,
                               cv2.VideoWriter_fourcc(*"mp4v"), 
                               desired_fps, 
                               #(640, 480))
                               desired_px)

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
    # view_in_counts=True,
    view_out_counts=False,
    view_in_counts=True   
)

runtime = 0.0
frame_count = 0

while True:
    # Read a frame from the webcam
    #ret, frame = cap.read()
    ret, im0 = cap.read()
    frame_count += 1
    
    tracks = model.track(im0, persist=True, show=False, conf=confidence)
    im0 = counter.start_counting(im0, tracks)
    
    video_writer.write(im0)

    # Display the frame
    #cv2.imshow('Webcam', im0)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
video_writer.release()
cv2.destroyAllWindows()

cukes_counted = max(counter.in_counts, counter.out_counts)

print("\nVideo processing has been successfully completed.")
print(f"Output video file: {outfilename}")
print(f"{cukes_counted} cucumbers counted in {frame_count} frames,")
print(f"i.e. {cukes_counted/(frame_count/desired_fps):.2f} cucumbers per second.\n")
