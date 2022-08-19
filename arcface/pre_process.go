package arcface

import (
	"image"
	"image/color"

	"github.com/disintegration/imaging"
)


func transposeRGB(rgbs []float32) []float32 {
	out := make([]float32, len(rgbs))
	channelLength := len(rgbs) / 3
	for i := 0; i < channelLength; i++ {
		out[i] = rgbs[i*3]
		out[i+channelLength] = rgbs[i*3+1]
		out[i+channelLength*2] = rgbs[i*3+2]
	}
	return out
}

// pre-process input data for face-detect model
func preprocessImage(src image.Image, inputSize int) ([]float32, float32) {
	var newHeight, newWidth int
	im_ratio := float32(src.Bounds().Dx()) / float32(src.Bounds().Dy())
	if im_ratio > 1 { // width > height
		newWidth = inputSize
		newHeight = int(float32(newWidth) / im_ratio)
	} else {
		newHeight = inputSize
		newWidth = int(float32(newHeight) * im_ratio)		
	}

	result := imaging.Resize(src, newWidth, newHeight, imaging.Lanczos)
	result = padBox(result)

	rgbs := make([]float32, inputSize*inputSize*3)

	j := 0
	for i := range result.Pix {
		if (i+1)%4 != 0 {
			rgbs[j] = float32(result.Pix[i])
			j++
		}
	}

	rgbs = transposeRGB(rgbs)

	channelLength := len(rgbs) / 3
	for i := 0; i < channelLength; i++ {
		rgbs[i] = normalize(rgbs[i], 127.5, 128.0)
		rgbs[i+channelLength] = normalize(rgbs[i+channelLength], 127.5, 128.0)
		rgbs[i+channelLength*2] = normalize(rgbs[i+channelLength*2], 127.5, 128.0)
	}

	return rgbs, float32(newHeight) / float32(src.Bounds().Dy())
}


func normalize(in float32, m float32, s float32) float32 {
	return (in - m) / s
}


// change image to square rect, padding with color Black
func padBox(src image.Image) *image.NRGBA {
	var maxW int

	if src.Bounds().Dx() > src.Bounds().Dy() {
		maxW = src.Bounds().Dx()
	} else {
		maxW = src.Bounds().Dy()
	}

	dst := imaging.New(maxW, maxW, color.Black)
	dst = imaging.Paste(dst, src, image.Point{0,0})

	//_ = imaging.Save(dst, "data/pading.jpg")

	return dst
}

// pre-process input data for face-features model
func preprocessFace(src image.Image, inputSize int) ([]float32) {
	result := padBox(src)

	rgbs := make([]float32, inputSize*inputSize*3)

	j := 0
	for i := range result.Pix {
		if (i+1)%4 != 0 {
			rgbs[j] = float32(result.Pix[i])
			j++
		}
	}

	rgbs = transposeRGB(rgbs)

	channelLength := len(rgbs) / 3
	for i := 0; i < channelLength; i++ {
		rgbs[i] = normalize(rgbs[i], 127.5, 127.5)
		rgbs[i+channelLength] = normalize(rgbs[i+channelLength], 127.5, 127.5)
		rgbs[i+channelLength*2] = normalize(rgbs[i+channelLength*2], 127.5, 127.5)
	}

	return rgbs
}
