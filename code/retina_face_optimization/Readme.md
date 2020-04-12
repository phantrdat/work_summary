1. Install TensorRT (Test with version 6)
2. Install torch2trt from https://github.com/NVIDIA-AI-IOT/torch2trt
3. Install pytorch 1.4
4. Run demo: python detect_tensorRT.py

Inferences stat (Test with NVidia V100 32Gb GPU )

| Sample Number | Input Size | Original model (ms) | Optimized model (ms) | Speed-up |
|:-------------:|:----------:|:-------------------:|:--------------------:|:--------:|
|      1000     |   224x224  |     7.394374909     |      0.647736741     |    11x   |
|      1000     |   480x480  |     14.70280075     |      1.994400769     |    7x    |
|      1000     |   600x600  |      22.034226      |      3.876773683     |    6x    |
|      1000     |   640x640  |     29.46028242     |      5.963048419     |    5x    |
|      1000     |   800x800  |     36.78949674     |      9.030155472     |    4x    |
|      1000     |  1280x1208 |     50.18559209     |      16.33359863     |    3x    |
|      1000     |   480x600  |     7.388626372     |      1.592058558     |    5x    |
|      1000     |  720x1280  |     15.13340619     |      5.852979702     |    3x    |
|      1000     |  1920x1080 |     32.29575615     |      14.96303666     |    2x    |
