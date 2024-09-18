from argparse import ArgumentParser
import numpy as np
import torch,os
from yoloworld import TextEmbedder



# Initialize text embedder
text_embedder = TextEmbedder(device="cpu")
text_token = text_embedder.tokenize(["person", "bicycle", "car", "motorcycle"])


torch.onnx.export(text_embedder, text_token, "models/yoloworld.vitb.txt.onnx")
os.system("onnxsim models/yoloworld.vitb.txt.onnx models/yoloworld.vitb.txt.onnx")


coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"]
    

os.makedirs("tokens", exist_ok=True)

coco_names_group4 = [coco_names[i:i+4] for i in range(0, len(coco_names), 4)]

for class_name_ in coco_names_group4:
    print(f"Saving {class_name_}")
    class_name = class_name_[0]

    # Get text embeddings
    text_token = text_embedder.tokenize(class_name_).cpu().numpy()

    np.save(f"tokens/{class_name.replace(' ', '_')}.npy", text_token)

os.system("tar -cvf yolo_world_calib_token_data.tar tokens/*.npy")
