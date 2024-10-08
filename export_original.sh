if [ ! -f "yolov8s-worldv2.pt" ]; then
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt
fi

if [ ! -d "third_party" ]; then
    mkdir third_party
fi
cd third_party
if [ ! -d "ultralytics" ]; then
    git clone https://github.com/ZHEQIUSHUI/ultralytics.git
    cd ultralytics
    git checkout no_einsum
    cd ..
fi
cd ../

if [ ! -d "ultralytics" ]; then
    ln -s third_party/ultralytics/ultralytics .
fi

cp yoloworld/ModelExporter_original.py yoloworld/ModelExporter.py

python export_ultralytics_model.py --img_height 640 --img_width 640 --num_classes 4 --model_name yolov8s-worldv2.pt 
onnxsim models/yolov8s-worldv2.onnx models/yolov8s-worldv2-original.onnx
