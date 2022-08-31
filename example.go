package main


import (
	"log"
	"github.com/disintegration/imaging"

	"github.com/jack139/arcface-go/arcface"
)

const (
	test_image_path = "diego.jpg"
)


func main() {
	if err := arcface.LoadOnnxModel("../../cv/face_model/arcface/models/buffalo_l"); err!=nil {
		log.Fatal("Load model fail: ", err.Error())
	}

	// load image
	srcImage, err := imaging.Open(test_image_path)
	if err != nil {
		log.Fatal("Open image error: ", err.Error())
	}

	dets, kpss, err := arcface.FaceDetect(srcImage)
	if err != nil {
		log.Fatal("FaceDetect() error: ", err.Error())
	}

	log.Println("face num: ", len(kpss))

	if len(dets)==0 {
		log.Println("No face detected.")
		return		
	}


	/*
	// crop face by detect boxes without normalization
	sr := image.Rectangle{
		image.Point{int(dets[0][0]), int(dets[0][1])}, 
		image.Point{int(dets[0][2]), int(dets[0][3])},
	}
	src2 := imaging.Crop(srcImage, sr)
	_ = imaging.Save(src2, "crop_face.jpg")
	*/


	// just use the first face data, which score is the highest
	features, normFace, err := arcface.FaceFeatures(srcImage, kpss[0])
	if err != nil {
		log.Fatal("FaceFeatures() error: %s\n", err.Error())
	}

	// normalized face image
	_ = imaging.Save(normFace, "norm_face.jpg")

	log.Println("features: ", features)
}
