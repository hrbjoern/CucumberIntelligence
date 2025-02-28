import cv2
from ultralytics import YOLO, solutions

#modelpath = '2024-09-28_1333.pt'
#modelpath = '2024-09-28_2146_segs_best.onnx'
modelpath = '2024-09-28_2145_boxes_best.onnx'

#videopath = "Gurkenvideo_short.mp4"
#videopath = "Data/FrischeGurken/VID_20240928_121553.mp4"
videopath = "Data/Videos/mah04777.mp4"
videopath = "Data/Videos/mah04331.mp4"
videopath = "Data/Videos/vlc-record-2024-09-30-12h00m16s-mah04331.mp4-.mp4"

#videopath = "/home/hrbjoern/Videos/Webcam/2024-09-29-133636.webm"



if modelpath == '2024-09-28_2146_segs_best.onnx':
    model = YOLO(modelpath, task='segment')
else:
    model = YOLO(modelpath, task='detect')

    
cap = cv2.VideoCapture(videopath)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

print("w, h, fps:", w, h, fps)

# Define line points
#line_points = [(20, 400), (1080, 400)]
if videopath == "Data/Videos/mah04777.mp4":
    line_points = [(240, 50), (1180, 50)]
elif videopath == "Data/Videos/mah04331.mp4" or videopath == "Data/Videos/vlc-record-2024-09-30-12h00m16s-mah04331.mp4-.mp4":
    line_points = [(1180, 50), (1180, 960)]

print("line_points:", line_points)
    
# Actually, try a region, too: 
region_points = [(240, 250), (1180, 250), (1180, 150), (240, 150)]

outfilename = "belt_counting_"+videopath+"_output.avi"
# Video writer
video_writer = cv2.VideoWriter(outfilename, 
                               cv2.VideoWriter_fourcc(*"mp4v"), 
                               fps, 
                               (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    #reg_pts=region_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
    view_in_counts=True,
    view_out_counts=False
    #view_out_counts=True
)

# For computing Gdot: 
frame_count = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    frame_count += 1
    print(f"Frame {frame_count}")

    tracks = model.track(im0, persist=True, show=False)
    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

cukes_counted = max(counter.in_counts, counter.out_counts)

print("Video processing has been successfully completed.")
print(f"Output video file: {outfilename}")
print(f"{cukes_counted} cucumbers counted in {frame_count} frames,")
print(f"i.e. {cukes_counted/(frame_count/fps):.2f} cucumbers per second.")