# Arcface-go

Go implementation of Arcface inference



## Prerequisites

- The onnx-format models used in the code is ["**buffalo_l**"](https://insightface.cn-sh2.ufileos.com/models/buffalo_l.zip) from [insightface](https://github.com/deepinsight/insightface/tree/master/model_zoo).
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (v1.12.1 in my environment) is required.
- [OpenCV](https://github.com/opencv/opencv) (v4.5.5 in my environment) is required, because some codes borrowed from [gocv](https://github.com/hybridgroup/gocv) to implement EstimateAffinePartial2DWithParams().



## Run example

The example is too simple, detect faces in the input image and retrieve features of the first face.

```
CGO_CPPFLAGS="-I/usr/local/include/opencv4" CGO_LDFLAGS="-lopencv_core -lopencv_calib3d -lopencv_imgproc" LD_LIBRARY_PATH=/usr/local/lib go run example.go
```

- If your ```onnxruntime``` and ```opencv``` is not installed in ```/usr/local```, some paths in CGO flags and ```LD_LIARARY_PATH``` should be corrected.
- path to "**buffalo_l**" should be corrected in ```example.go``` .



## Sample API
[arcface-go-api](https://github.com/jack139/arcface-go-api)
