from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv
import os 
load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("leaf-microplastics").project("leaaaaaaf")
version = project.version(3)
dataset = version.download("yolov8", location="./data")

path = dataset.location + "/data.yaml"
print(path)

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

#Use the model
results = model.train(data=path, epochs=15, verbose=True, optimizer='Adam', imgsz=800)  # train the model

# Save the trained model
results = model.val()  # evaluate model performance on the validation set
results = model.export()  # export the model