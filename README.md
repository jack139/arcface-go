# Arcface-go

Go implementation of Arcface inference



## Prerequisites

- The onnx-format models used the code is ["**buffalo_l**"](https://insightface.cn-sh2.ufileos.com/models/buffalo_l.zip) from [insightface](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- OpenCV (v4.5.5 in my environment) is needed, because some codes borrowed from [gocv](https://github.com/hybridgroup/gocv) to implement EstimateAffinePartial2DWithParams().



## Run example

The example is too simple, detect faces in the input image and retrieve features of the first face.

```
CGO_LDFLAGS="-lopencv_core -lopencv_calib3d -lopencv_imgproc" LD_LIBRARY_PATH=/usr/local/lib go run example.go
```

- If your opencv is not installed in ```/usr/local```, some paths in CGO flags and LD_LIARARY_PATH should be corrected.
- path to "**buffalo_l**" should be corrected in ```example.go ```
