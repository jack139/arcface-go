package gocvx

/*
#cgo !windows pkg-config: opencv4
#cgo CXXFLAGS:   --std=c++11
#cgo !windows CPPFLAGS: -I/usr/local/include/opencv4
#cgo !windows LDFLAGS: -L/usr/local/lib -lopencv_core -lopencv_calib3d -lopencv_imgproc
*/

import "C"