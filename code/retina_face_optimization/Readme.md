1. Install TensorRT (Test with version 6)
2. Install torch2trt from https://github.com/NVIDIA-AI-IOT/torch2trt
3. Install pytorch 1.4
4. Run demo: python detect_tensorRT.py

Inferences stat (Test with NVidia V100 32Gb GPU )

| RetinaFace (face detection) optimization statistics |            |                         |                 |          |
|-----------------------------------------------------|------------|-------------------------|-----------------|----------|
| Sample Number                                       | Input Size | Inference Time (second) |                 | Speed-up |
|                                                     |            | Original model          | Optimized model |          |
| 1000                                                | 224x224    | 0.007394375             | 0.000647737     | 11x      |
| 1000                                                | 480x480    | 0.014702801             | 0.001994401     | 7x       |
| 1000                                                | 600x600    | 0.022034226             | 0.003876774     | 6x       |
| 1000                                                | 640x640    | 0.029460282             | 0.005963048     | 5x       |
| 1000                                                | 800x800    | 0.036789497             | 0.009030155     | 4x       |
| 1000                                                | 1280x1208  | 0.050185592             | 0.016333599     | 3x       |
| 1000                                                | 480x600    | 0.007388626             | 0.001592059     | 5x       |
| 1000                                                | 720x1280   | 0.015133406             | 0.00585298      | 3x       |
| 1000                                                | 1920x1080  | 0.032295756             | 0.014963037     | 2x       |
