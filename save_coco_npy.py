from argparse import ArgumentParser
import numpy as np
import os

from yoloworld import TextEmbedder

coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"]


# Initialize text embedder
text_embedder = TextEmbedder(device="cpu")

os.makedirs("tmp", exist_ok=True)

coco_names_group4 = [coco_names[i:i+4] for i in range(0, len(coco_names), 4)]

print(coco_names_group4)

for class_name_ in coco_names_group4:
    print(f"Saving {class_name_}")
    class_name = class_name_[0]

    # Get text embeddings
    class_embeddings = text_embedder.embed_text(class_name_)

    # Convert to numpy array
    class_embeddings = class_embeddings.cpu().numpy().astype(np.float32)
    
    np.savez(f"tmp/{class_name.replace(' ', '_')}.npz", class_embeddings=class_embeddings, class_list=np.array(class_name_))
    np.save(f"tmp/{class_name.replace(' ', '_')}.npy", class_embeddings)
    with open(f"tmp/{class_name.replace(' ', '_')}.bin", "wb") as f:
        f.write(class_embeddings.tobytes())
