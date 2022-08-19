# Arcface-go

Go re-implementation of Arcface inference



## Run example
```
CGO_LDFLAGS="-L/usr/local/lib -lopencv_core -lopencv_calib3d -lopencv_imgproc" go build -o data/
LD_LIBRARY_PATH=/usr/local/lib data/arcface-go
```