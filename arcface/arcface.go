package arcface

import (
	"errors"
	"path/filepath"
	"image"

	"github.com/ivansuteja96/go-onnxruntime"
	//"github.com/disintegration/imaging"
)

const (
	det_model_input_size = 224
	face_align_image_size = 112
)

var (
	detModel *onnxruntime.ORTSession
	arcfaceModel *onnxruntime.ORTSession
)

func LoadOnnxModel(onnxmodel_path string) (err error) {
	ortEnvDet := onnxruntime.NewORTEnv(onnxruntime.ORT_LOGGING_LEVEL_ERROR, "development")
	ortDetSO := onnxruntime.NewORTSessionOptions()

	detModel, err = onnxruntime.NewORTSession(ortEnvDet, filepath.Join(onnxmodel_path, "det_10g.onnx"), ortDetSO)
	if err != nil {
		return err
	}

	arcfaceModel, err = onnxruntime.NewORTSession(ortEnvDet, filepath.Join(onnxmodel_path, "w600k_r50.onnx"), ortDetSO)
	if err != nil {
		return err
	}

	return nil
}

// Detect face in src image
func FaceDetect(src image.Image) ([][]float32, [][]float32, error) {
	shape1 := []int64{1, 3, det_model_input_size, det_model_input_size}
	input1, det_scale := preprocessImage(src, det_model_input_size)

	// face detect model inference
	res, err := detModel.Predict([]onnxruntime.TensorValue{
		{
			Value: input1,
			Shape: shape1,
		},
	})
	if err != nil {
		return nil, nil, err
	}

	if len(res) == 0 {
		return nil, nil, errors.New("Fail to get result")
	}

	dets, kpss := processResult(res, det_scale)

	return dets, kpss, nil
}

// Get face features by Arcface
// src is original image
// lmk is face landmark detected by FaceDetect()
func FaceFeatures(src image.Image, lmk []float32) ([]float32, error) {
	aimg, err := norm_crop(src, lmk)
	if err!=nil {
		return nil, err
	}

	// save normalization crop face for test
	//_ = imaging.Save(aimg, "data/crop_norm.jpg")

	// prepare input data
	shape2 := []int64{1, 3, face_align_image_size, face_align_image_size}
	input2 := preprocessFace(aimg, face_align_image_size)


	// face features modle inference
	res2, err := arcfaceModel.Predict([]onnxruntime.TensorValue{
		{
			Value: input2,
			Shape: shape2,
		},
	})
	if err != nil {
		return nil, err
	}

	if len(res2) == 0 {
		return nil, errors.New("Fail to get result")
	}

	return res2[0].Value.([]float32), nil
}