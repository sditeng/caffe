#include <vector>
#include <cfloat>

#include "caffe/layer.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a, const Dtype* bottom_data_b,
  const Dtype* batch_coeff, const int data_blob_idx, const int num_batches,
  const int inner_size, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int batch_idx = index/inner_size;
    // do nothing if current batch_coeff is 0
    if (batch_coeff != NULL && batch_coeff[data_blob_idx*num_batches + batch_idx] == Dtype(0))
      return;
    if (bottom_data_a[index] < bottom_data_b[index]) {
      top_data[index] = bottom_data_b[index];
    }
  }
}

template <typename Dtype>
__global__ void SumForward(const int nthreads, const Dtype* bottom_data,
  const Dtype* batch_coeff, const int data_blob_idx, const int num_batches,
  const int inner_size, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int batch_idx = index/inner_size;
    // do nothing if current batch_coeff is 0
    top_data[index] += bottom_data[index] *
      (batch_coeff != NULL ? batch_coeff[data_blob_idx*num_batches + batch_idx]: Dtype(1.0));
  }
}

template <typename Dtype>
void MergeBatchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  int start_data_blob = ignore_blob_ ? 1 : 0;
  int inner_size = bottom[start_data_blob]->count(1);
  int count = top[0]->count();

  // set batch_coeff from bottom[0] if given
  const Dtype* batch_coeff = NULL;
  if (ignore_blob_){
    batch_coeff = bottom[0]->gpu_data();
  }

  switch (op_) {
    case MergeBatchParameter_MergeOp_SUM:
      // Initialize
      caffe_gpu_set(count, Dtype(0.0), top_data);
      for (int i = start_data_blob; i < bottom.size(); ++i) {
        const Dtype* bottom_data_i = bottom[i]->gpu_data();
        // need to substract ignore_blob
        SumForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data_i, batch_coeff, i-start_data_blob, num_batches_, inner_size, top_data);
      }
    break;
    case MergeBatchParameter_MergeOp_MAX:
      // Initialize
      caffe_gpu_set(count, Dtype(-FLT_MAX), top_data);
      for (int i = start_data_blob; i < bottom.size(); ++i) {
        const Dtype* bottom_data_i = bottom[i]->gpu_data();
        // NOLINT_NEXT_LINE(whitespace/operators)
        MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, top_data, bottom_data_i, batch_coeff, i-start_data_blob, num_batches_, inner_size, top_data);
      }
    break;
    default:
        LOG(FATAL) << "Unknown merge operation.";
    }
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff, const Dtype* top_data,
    const Dtype* bottom_data, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  // equivalent to MIL, put diff to the bottom blob that is equal to top_data
    Dtype top_val = top_data[index];
    if (top_val == bottom_data[index] && top_val != -FLT_MAX)
      bottom_diff[index] = top_diff[index];
    }
}

template <typename Dtype>
__global__ void SumBackward( const int nthreads, const Dtype* top_diff, const Dtype* batch_coeff,
  const int data_blob_idx, const int num_batches, const int inner_size, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int batch_idx = index/inner_size;
    // bottom diff is scaled top diff
    bottom_diff[index] = top_diff[index] *
      (batch_coeff!=NULL?batch_coeff[data_blob_idx*num_batches + batch_idx]:Dtype(1.0));
  }
}


template <typename Dtype>
void MergeBatchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int start_data_blob = ignore_blob_ ? 1 : 0;
  const Dtype* top_diff=top[0]->gpu_diff();
  const Dtype* top_data=top[0]->gpu_data();
  int inner_size = bottom[start_data_blob]->count(1);
  int count = bottom[start_data_blob]->count();

  // set batch_coeff from bottom[0] if given
  const Dtype* batch_coeff = NULL;
  if (ignore_blob_){
    batch_coeff = bottom[0]->gpu_data();
  }

  for (int i=start_data_blob; i<bottom.size(); i++){
    if(propagate_down[i]) {
      Dtype* bottom_diff_i = bottom[i]->mutable_gpu_diff();
      caffe_gpu_set(count, Dtype(0.0), bottom_diff_i);
      const Dtype* bottom_data_i = bottom[i]->gpu_data();
      switch (op_) {
        case MergeBatchParameter_MergeOp_SUM:
          // LOG(INFO) << "SUM";
          SumBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, batch_coeff, i-start_data_blob, num_batches_, inner_size, bottom_diff_i);
          break;
        case MergeBatchParameter_MergeOp_MAX:
          // LOG(INFO) << "MAX";
          MaxBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top_data, bottom_data_i, bottom_diff_i);
          break;
        case MergeBatchParameter_MergeOp_MEAN:
          break;
        default:
          LOG(FATAL) << "Unknown merge operation.";
        }
      }
    }
  }


INSTANTIATE_LAYER_GPU_FUNCS(MergeBatchLayer);

}  // namespace caffe
