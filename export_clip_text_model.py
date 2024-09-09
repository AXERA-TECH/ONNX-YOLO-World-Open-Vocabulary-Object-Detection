from argparse import ArgumentParser
import numpy as np
import torch,os
from yoloworld import TextEmbedder



# Initialize text embedder
text_embedder = TextEmbedder(device="cpu")

text_token = text_embedder.tokenize("cat")

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

prefixs = ["",
          "a ","an ",
          "a picture of ", "a picture of a ", "a picture of an ",
          "a image of ", "a image of a ", "a image of an ",
          "a photo of ", "a photo of a ", "a photo of an ",
          ]


os.makedirs("tokens", exist_ok=True)

for class_name_ in coco_names:
    
    for prefix in prefixs:
        print(f"Saving {prefix}{class_name_}")
        class_name = prefix + class_name_

        # Get text embeddings
        text_token = text_embedder.tokenize(class_name).cpu().numpy()

        np.save(f"tokens/{class_name.replace(' ', '_')}.npy", text_token)

os.system("tar -cvf yolo_world_calib_token_data.tar tokens/*.npy")


