package arcface

import (
	"fmt"
	"sort"

	"github.com/ivansuteja96/go-onnxruntime"
)

const (
	nms_thresh = float32(0.4)
	det_thresh = float32(0.5)
)

var (
	// len(outputs)==9
	_fmc = 3
	_feat_stride_fpn = []int{8, 16, 32}
	_num_anchors = 2
)

// process result after face-detect model inferenced
func processResult(net_outs []onnxruntime.TensorValue, det_scale float32) ([][]float32, [][]float32) {
	//for i:=0;i<len(net_outs);i++ {
	//	log.Printf("Success do predict, shape : %+v, result : %+v\n", 
	//		net_outs[i].Shape, 
	//		net_outs[i].Value.([]float32)[:net_outs[i].Shape[1]], // only show one value
	//	)
	//}

	center_cache := make(map[string][][]float32)

	var scores_list []float32
	var bboxes_list [][]float32
	var kpss_list [][]float32

	for idx := range _feat_stride_fpn {
		stride := _feat_stride_fpn[idx]
		scores := net_outs[idx].Value.([]float32)
		bbox_preds := net_outs[idx+_fmc].Value.([]float32)
		for i := range bbox_preds { 
			bbox_preds[i] = bbox_preds[i] * float32(stride)
		}

		var kps_preds []float32 // landmark
		kps_preds = net_outs[idx+_fmc*2].Value.([]float32)
		for i := range kps_preds { 
			kps_preds[i] = kps_preds[i] * float32(stride)
		}

		height := det_model_input_size / stride
		width := det_model_input_size / stride
		key := fmt.Sprintf("%d-%d-%d", height, width, stride)
		var anchor_centers [][]float32
		if val, ok := center_cache[key]; ok {
			anchor_centers = val
		} else {
			anchor_centers = make([][]float32, height*width*_num_anchors)
			for i:=0;i<height;i++ {
				for j:=0;j<width;j++ {
					for k:=0;k<_num_anchors;k++ {
						anchor_centers[i*width*_num_anchors+j*_num_anchors+k] = []float32{float32(j*stride), float32(i*stride)}
					}
				}
			}
			//log.Println(stride, len(anchor_centers), anchor_centers)

			if len(center_cache)<100 {
				center_cache[key] = anchor_centers
			}		
		}

		// filter by det_thresh == 0.5
		var pos_inds []int
		for i := range scores {
			if scores[i]>det_thresh {
				pos_inds = append(pos_inds, i)
			}
		}

		bboxes := distance2bbox(anchor_centers, bbox_preds)
		kpss := distance2kps(anchor_centers, kps_preds)

		for i:=range pos_inds {
			scores_list = append(scores_list, scores[pos_inds[i]])
			bboxes_list = append(bboxes_list, bboxes[pos_inds[i]])
			kpss_list = append(kpss_list, kpss[pos_inds[i]])
		}
	}


	// post process after get boxes and landmarks

	for i := range bboxes_list {
		for j:=0;j<4;j++ {
			bboxes_list[i][j] /= det_scale
		}
		bboxes_list[i] = append(bboxes_list[i], scores_list[i])

		for j:=0;j<10;j++ {
			kpss_list[i][j] /= det_scale
		}
		kpss_list[i] = append(kpss_list[i], scores_list[i])
	}

	sort.Slice(bboxes_list, func(i, j int) bool { return bboxes_list[i][4] > bboxes_list[j][4] })
	sort.Slice(kpss_list, func(i, j int) bool { return kpss_list[i][10] > kpss_list[j][10] })

	keep := nms(bboxes_list)

	det := make([][]float32, len(keep))
	kpss := make([][]float32, len(keep))
	for i := range keep {
		det[i] = bboxes_list[keep[i]]
		kpss[i] = kpss_list[keep[i]]
	}

	return det, kpss
}


func distance2bbox(points [][]float32, distance []float32) (ret [][]float32) {
	ret = make([][]float32, len(points))
	for i := range points {
		ret[i] = []float32{
			points[i][0] - distance[i*4+0],
			points[i][1] - distance[i*4+1],
			points[i][0] + distance[i*4+2],
			points[i][1] + distance[i*4+3],
		}
	}
	return
}

func distance2kps(points [][]float32, distance []float32) (ret [][]float32) {
	ret = make([][]float32, len(points))
	for i := range points {
		ret[i] = make([]float32, 10)
		for j:=0;j<10;j=j+2 {
			ret[i][j]   = points[i][j%2] + distance[i*10+j]
			ret[i][j+1] = points[i][j%2+1] + distance[i*10+j+1]
		} 
	}
	return
}


func max(a, b float32) float32 {
	if a>b { 
		return a
	} else {
		return b
	}
}

func min(a, b float32) float32 {
	if a<b { 
		return a
	} else {
		return b
	}
}

func nms(dets [][]float32) (ret []int) {
	if len(dets)==0 {
		return
	}

	var order []int
	areas := make([]float32, len(dets))
	for i := range dets {
		order = append(order, i)
		areas[i] = (dets[i][2] - dets[i][0] + 1) * (dets[i][3] - dets[i][1] + 1)
	}
	for len(order)>0 {
		i := order[0]
		ret = append(ret, i)

		var keep []int
		for j := range order[1:] {
			xx1 := max(dets[i][0], dets[order[j+1]][0])
			yy1 := max(dets[i][1], dets[order[j+1]][1])
			xx2 := min(dets[i][2], dets[order[j+1]][2])
			yy2 := min(dets[i][3], dets[order[j+1]][3])

			w := max(0.0, xx2 - xx1 + 1)
			h := max(0.0, yy2 - yy1 + 1)
			inter := w * h
			ovr := inter / (areas[i] + areas[order[j+1]] - inter)

			if ovr <= nms_thresh {
				keep = append(keep, order[j+1])
			}
		}

		order = keep
	}
	return
}
