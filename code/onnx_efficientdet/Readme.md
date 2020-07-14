For GPU: 
- pip install onnxruntime-gpu

For CPU:
- pip install onnxruntime

Install protobuf: 

- pip install protobuf==3.12.2

Convert pytorch model from [https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

Run: python onnx_inference.py --img <path_to_img> --onnx_model <path_to_model> --compound_coef <compound_coefficient>
