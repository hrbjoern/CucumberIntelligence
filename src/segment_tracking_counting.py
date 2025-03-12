from collections import defaultdict

import cv2
import sys, datetime

from ultralytics import YOLO, solutions
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(lambda: [])

#model = YOLO("yolov8n-seg.pt")  # segmentation model
#modelpath = '2024-09-28_2146_segs_best.onnx'
modelpath = 'Models/2024-09-28_2146_segs_best.pt'
model = YOLO(modelpath, task='segment')

#cap = cv2.VideoCapture("Data/FrischeGurken/VID_20240928_121553.mp4")
#videopath = "Data/Videos/mah04777.mp4"
videoname = "mah04777.mp4"
videopath = "RawVideos/"+videoname


# Confidence threshold: 
confidence = 0.3

print(f"Input video file: {videopath}")
print(f"Confidence threshold: {confidence}")

# Output video file name:
from datetime import date
datestr = date.today().strftime("%Y-%m-%d")+"_"

outfilename = "FinalResults/"+datestr+videoname+"_seg-instance-counting.avi"



cap = cv2.VideoCapture(videopath)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(outfilename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_points = [(240, 50), (1180, 50)]
if "mah04777" in videopath:
    line_points = [(240, 50), (1180, 50)]
elif "mah04331" in videopath:
    line_points = [(1180, 50), (1180, 960)]
elif "133636" in videopath:
    line_points = [(400, 50), (400, 450)]
else: 
    print("Video unknown, please specify line_points!")
    sys.exit()

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
    # view_in_counts=False,
    # view_out_counts=True
    view_in_counts=True,
    view_out_counts=False
)


# For computing Gdot: 
frame_count = 0

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Frame {frame_count}")

    annotator = Annotator(im0, line_width=2)
    results = model.track(im0, persist=True, conf=confidence)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            if mask.sum() == 0:
                continue

            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, 
                               label=str(track_id), txt_color=txt_color)

            # annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), 
            #                    label=str(track_id)
            #                    )
            
    im0 = counter.start_counting(im0, results)

    out.write(im0)
    #cv2.imshow("instance-segmentation-object-tracking", im0)
    
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

cukes_counted = max(counter.in_counts, counter.out_counts)

print("\nVideo processing has been successfully completed.")
print(f"Output video file: {outfilename}")
print("(make sure to convert avi to mp4 if it's too big)\n")
print(f"{cukes_counted} cucumbers counted in {frame_count} frames,")
print(f"i.e. {cukes_counted/(frame_count/fps):.2f} cucumbers per second.\n")