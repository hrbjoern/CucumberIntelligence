from ultralytics import YOLO

#model = YOLO("/home/hrbjoern/Desktop/2024-09_CucumberIntelligence/2024-09-27_abends_setup002.pt")
#model = YOLO("2024-09-28_1333.pt")
#model = YOLO("2024-09-28_2145_boxes_best.pt")
#model = YOLO("2024-09-28_2145_boxes_best.onnx")
model = YOLO("2024-09-28_2146_segs_best.onnx", task='segment')

imagesize = 640
confidence = 0.7


# Test the model on a few images:
results = model.predict(["20220614_134218_Gurkenfoto_2.jpg",
                         "131055.jpg",
                         "20220614_134218_Gurkenfoto_rescaled_640_segments.jpg",
                         "20220614_134218_Gurkenfoto.jpg", 
                         "Data/FrischeGurken/IMG_20240928_121633.jpg", 
                         "Data/FrischeGurken/IMG_20240928_121631.jpg"],
                        imgsz=imagesize, 
                        conf=confidence,
                        #show=True,
                        #save=True)
                        save=False)

for result in results:
  boxes = result.boxes
  probs = result.probs
  masks = result.masks
  result.show()

#print(boxes, probs)