from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("/content/drive/My Drive/Firenet/detection_model-ex-33--loss-4.97.h5") 
detector.setJsonPath("/content/drive/My Drive/Firenet/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="1.jpg", output_image_path="img-detected.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
