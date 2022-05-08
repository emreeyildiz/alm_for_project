from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from google.cloud import storage
from collections import OrderedDict
from operator import getitem
import cv2

class Image(BaseModel):
    image_name: str

class Prediction(BaseModel):
    predictions: dict

app = FastAPI()
'''
storage_client = storage.Client()
'''
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
mappings = {
    "Car": [("Road Problem", 0.5), ("Transportation", 0.3), ("Accident", 0.1), ("Road Maintenance", 0.1)],
    "Cat": [("Stray Animals", 0.8), ("Volunteering", 0.2)],
    "Dog": [("Stray Animals", 0.8), ("Volunteering", 0.2)],
    "Garbage": [("Garbage", 0.8), ("Waste", 0.2)],
    "Human-face": [("Missing", 0.4), ("Volunteering", 0.6)],
    "Person": [("Road Problem", 0.2), ("Missing", 0.4), ("Volunteering", 0.5), ("Accident", 0.1)],
    "Pothole": [("Road Problem", 0.5), ("Transportation", 0.1), ("Accident", 0.1), ("Road Maintenance", 0.3)],
    "Street-light": [("Road Problem", 0.4), ("Electricity", 0.3), ("Transportation", 0.2), ("Accident", 0.1)],
    "Traffic-light": [("Road Problem", 0.4), ("Transportation", 0.2), ("Accident", 0.4)],
    "Traffic-sign": [("Road Problem", 0.4), ("Transportation", 0.1), ("Accident", 0.1), ("Road Maintenance", 0.4)],
    "bus": [("Road Problem", 0.3), ("Transportation", 0.6), ("Accident", 0.1)],
    "cyclist": [("Road Problem", 0.5), ("Transportation", 0.4), ("Accident", 0.1)],
    "pedestrian": [("Road Problem", 0.4), ("Transportation", 0.1), ("Missing", 0.3), ("Accident", 0.2)],
    "vehicle": [("Road Problem", 0.5), ("Transportation", 0.3), ("Accident", 0.1), ("Road Maintenance", 0.1)]
}
'''
bucket = storage_client.bucket(bucket_name)
'''

def predict_category():
    confidences = {}
    try:
        '''
        destination_file_name = "/tmp/" + req.image_name
        blob = bucket.blob("darknet/" + req.image_name)
        blob.download_to_filename(destination_file_name)
        '''
        img = cv2.imread("/Users/ardaakcabuyuk/Desktop/Year4/Year4Semester1/Senior/ml_service/app/people.jpeg")

        with open('app/obj.names', 'r') as f:
            classes = f.read().splitlines()

        net = cv2.dnn.readNetFromDarknet('app/reportown-yolo.cfg', 'app/reportown.v2.weights')

        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

        classIds, scores, boxes = model.detect(img, confThreshold=0.005, nmsThreshold=0.005)
        '''
        if classIds != ():
            classes = [classes[id] for id in classIds.tolist()]
            for pred in classes:
                conf_scores = mappings[pred]
                for (category, score) in conf_scores:
                    if category not in confidences:
                        confidences[category] = {"confidence": score, "objects": [{"class": pred, "count": 1}]}
                    else:
                        confidences[category]["confidence"] += score
                        objects = confidences[category]["objects"]
                        incremented = False
                        for object in objects:
                            if object["class"] == pred:
                                object["count"] += 1
                                incremented = True
                        if incremented == False:
                            objects.append({"class": pred, "count": 1})
        confidences = OrderedDict(sorted(confidences.items(), key = lambda x: getitem(x[1], 'confidence'))[::-1])
        '''
        for (classId, score, box) in zip(classIds, scores, boxes):
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                          color=(0, 255, 0), thickness=2)

            text = '%s: %.2f' % (classes[classId], score)
            cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        color=(0, 255, 0), thickness=1)

        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)


predict_category()
