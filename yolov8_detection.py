from ultralytics import YOLO
import torch.optim as optim
from roboflow import Roboflow

rf = Roboflow(api_key="whoopsies")
project = rf.workspace("leaf-microplastics").project("leaaaaaaf")
version = project.version(3)
dataset = version.download("yolov8")

path = dataset.location + "/data.yaml"

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Initialize the Adam optimizer
#adam = optim.Adam(model.parameters(), lr=0.01) # set learning rate

#Use the model
results = model.train(data=path, epochs=15, verbose=True, optimizer='Adam', imgsz=800)  # train the model

# Save the trained model
results = model.val()  # evaluate model performance on the validation set
results = model.export()  # export the model


# results = model('C:/Users/ekine/OneDrive/Desktop/CSCI/masters/ComputerVision/project/test_data/detection/images/dataset1_back_1.png')  # predict on an image
# # Load YOLOv8n-seg, train it on COCO128-seg for 3 epochs and predict an image with it
# from ultralytics import YOLO

# model = YOLO('yolov8n-seg.pt')  # load a pretrained YOLOv8n segmentation model
# model.train(data='apple_v8.yaml', epochs=3)  # train the model
# model('C:/Users/ekine/OneDrive/Desktop/CSCI/masters/ComputerVision/project/test_data/detection/images/dataset1_back_1.png')  # predict on an image
# results = model.export(format='onnx')  # export the model to ONNX format
