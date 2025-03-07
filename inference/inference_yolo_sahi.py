from sahi.utils.yolov8 import (
    download_yolov8s_model, download_yolov8s_seg_model
)

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
from pathlib import Path
import json

class SahiInference:
    def __init__(self, old_img_path, new_img_path, yolo_path):
        self.yolov8_model_path = yolo_path
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=yolo_path,
            confidence_threshold=0.3,
            device="cpu", # or 'cuda:0'
        )
        self.old_images_path = old_img_path
        self.new_images_path = new_img_path
        self.result = None
        self.final_prediction_data = {}

    def predict(self):
        files = Path(self.old_images_path).glob('*')
        for file in files:
            self.result = get_sliced_prediction(
                str(file),
                self.detection_model,
                slice_height = 256,
                slice_width = 256,
                overlap_height_ratio = 0.2,
                overlap_width_ratio = 0.2
            )
            new_file_name = str(file)[8:]
            self.result.export_visuals(export_dir=self.new_images_path, file_name=new_file_name)
            object_prediction_list = self.result.object_prediction_list
            predict_data = []
            i = 0
            for predictions in object_prediction_list:
                predict_data.append({"maxx": str(predictions.bbox.maxx), "maxy": str(predictions.bbox.maxy), "minx": str(predictions.bbox.minx), "miny": str(predictions.bbox.miny), "area": str(predictions.bbox.area), "prediction confidence": str(predictions.score.value)})
                i+=1
            self.final_prediction_data[new_file_name] = predict_data

    def save_to_file(self):
        out_file = open("final_output.json", "w")
        json.dump(self.final_prediction_data, out_file, indent = 4, sort_keys=True)
        out_file.close()


def main():
    inference = SahiInference("old_img/", "new_img", "best.pt")
    download_yolov8s_model(inference.yolov8_model_path)
    inference.predict()
    inference.save_to_file()


main()

