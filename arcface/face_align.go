package arcface

import (
	"fmt"
	"image"

	"arcface-go/gocvx"
)

var (
	// arcface matrix from insightface/utils/face_align.py
	arcface_src = []gocvx.Point2f{
	   {38.2946, 51.6963},
       {73.5318, 51.5014},
       {56.0252, 71.7366},
       {41.5493, 92.3655},
       {70.7299, 92.2041},
   }
)


// Crop face image and normalization
func norm_crop(srcImage image.Image, lmk []float32) (image.Image, error) {
	// similarity transform
	m := estimate_norm(lmk)
	defer m.Close()

	// print out the 2×3 transformation matrix
	//printM(m)

	// transfer to Mat
	src, err := gocvx.ImageToMatRGB(srcImage)
	if err!=nil {
		return nil, err
	}

	dst := src.Clone()
	defer dst.Close()

	// affine transformation to an image (Mat)
	gocvx.WarpAffine(src, &dst, m, image.Point{face_align_image_size, face_align_image_size})

	// Mat transfer to image
	aimg, err := dst.ToImage()
	if err!=nil {
		return nil, err
	}

	return aimg, nil
}


// equal to python: skimage.transform.SimilarityTransform()
func estimate_norm(lmk []float32) gocvx.Mat {
	dst := make([]gocvx.Point2f, 5)
	for i:=0;i<5;i++ {
		dst[i] = gocvx.Point2f{lmk[i*2], lmk[i*2+1]}
	}

	pvsrc := gocvx.NewPoint2fVectorFromPoints(arcface_src)
	defer pvsrc.Close()

	pvdst := gocvx.NewPoint2fVectorFromPoints(dst)
	defer pvdst.Close()

	inliers := gocvx.NewMat()
	defer inliers.Close()
	method := 4 // cv2.LMEDS
	ransacProjThreshold := 3.0
	maxiters := uint(2000)
	confidence := 0.99
	refineIters := uint(10)

	m := gocvx.EstimateAffinePartial2DWithParams(pvdst, pvsrc, inliers, method, 
												 ransacProjThreshold, maxiters, confidence, refineIters)

	return m
}

// print matrix, for test
func printM(m gocvx.Mat) {
	for i:=0;i<m.Rows();i++ {
		for j:=0;j<m.Cols();j++ {
			fmt.Printf("%v ", m.GetDoubleAt(i, j))
		}
		fmt.Printf("\n")
	}
}