import cv2
# from imread_from_url import imread_from_url
from yoloworld import YOLOWorld, DetectionDrawer, read_class_embeddings, TextEmbedder
import onnxruntime
import numpy as np

model_path = "./models/yolov8s-worldv2-original.onnx"

# get class embeddings
text_embedder = TextEmbedder(device="cpu")
class_list = ["person", "horse", "car", "dog"]
text_token = text_embedder.tokenize(class_list)
np.save("./yoloworld_onboard/demo_text_token_onboard.npy",text_token)
clip_path = "./models/yoloworld.vitb.txt.b1.onnx"
clip_session = onnxruntime.InferenceSession(clip_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
class_embeddings = [clip_session.run(['2202'],{'text_token': np.array([t])})[0][0][0] for t in text_token]
class_embeddings = np.array([class_embeddings])

# Initialize YOLO-World object detector
yoloworld_detector = YOLOWorld(model_path, conf_thres=0.3, iou_thres=0.5)

# Initialize DetectionDrawer
drawer = DetectionDrawer(np.array(class_list))

img_url = "./doc/img/ssd_horse.jpg"
img = cv2.imread(img_url)

# Detect Objects
boxes, scores, class_ids = yoloworld_detector(img, class_embeddings)

# Draw detections
combined_img = drawer(img, boxes, scores, class_ids)

'''
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", combined_img)
cv2.waitKey(0)
'''
cv2.imwrite("./doc/img/ssd_horse_result_with_clip.png", combined_img)
