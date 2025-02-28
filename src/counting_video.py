import cv2
from ultralytics import YOLO, solutions

#modelpath = '2024-09-28_1333.pt'
#modelpath = '2024-09-28_2146_segs_best.onnx'
modelpath = '2024-09-28_2145_boxes_best.onnx'

#videopath = "Gurkenvideo_short.mp4"
#videopath = "Data/FrischeGurken/VID_20240928_121553.mp4"
videopath = "Data/Videos/mah04777.mp4"
#videopath = "/home/hrbjoern/Videos/Webcam/2024-09-29-133636.webm"



if modelpath == '2024-09-28_2146_segs_best.onnx':
    model = YOLO(modelpath, task='segment')
else:
    model = YOLO(modelpath, task='detect')

    
cap = cv2.VideoCapture(videopath)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
#line_points = [(20, 400), (1080, 400)]
if videopath == "Data/Videos/mah04777.mp4":
    line_points = [(240, 50), (1180, 50)]
if videopath == "Data/FrischeGurken/VID_20240928_121553.mp4":
    line_points = [(20, 50), (20, 600)]
# else: 
#     line_points = [(400, 50), (400, 600)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", 
                               cv2.VideoWriter_fourcc(*"mp4v"), 
                               fps, 
                               (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
    view_in_counts=True,
    view_out_counts=False
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()