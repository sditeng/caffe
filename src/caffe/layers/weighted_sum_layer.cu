#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
  __global__ void forwardGPU(const int nthreads, const Dtype* bottom_data, const Dtype* weight,
    const Dtype* bias, int inner_size, int K0, int group, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // index = (outer_idx * K0 + channel) * inner_size_ + inner_idx
      // bottom_index = (outer_idx * group * K0 + i*K0 + channel) * inner_size_ + inner_idx
      // nthreads = outer_num * channel_out_ * inner_size_
      int channel = (index/inner_size) % K0;
      int outer_idx = (index/inner_size) / K0;
      bottom_data += outer_idx*group*K0*inner_size;
      for (int i = 0; i < group; ++i)
      {
        top_data[index] += weight[channel+i*K0] * bottom_data[channel+inner_size*K0*i];
      }
      if (bias != NULL)
        top_data[index] += bias[channel];
    }
  }

template <typename Dtype>
  void WeightedSumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    const Dtype* bias = NULL;
    if (bias_term_) {
     bias = this->blobs_[1]->gpu_data();
    }
    caffe_gpu_set(count, Dtype(0), top_data);
  // The first "axis_" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
   forwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data, weight, bias, inner_size_, channel_out_, group_, top_data);
 }

template <typename Dtype>
 __global__ void gpu_backward_bias(const int nthreads, const Dtype* top_diff,
  int inner_size, int outer_num, Dtype* bias_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      // top_index = (outer_idx * nthreads + index) * inner_size_ + inner_idx
      // nthreads = channel_out_
    int offset = nthreads*inner_size;
    for (int i = 0; i < outer_num; ++i)
    {
      for (int j = 0; j < inner_size; ++j)
        bias_diff[index] += top_diff[i*offset+index*inner_size+j];
    }
  }
}
template <typename Dtype>
 __global__ void gpu_backward_weight(const int nthreads, const Dtype* top_diff, const Dtype* bottom_data,
  int inner_size, int outer_num, int group, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      // top_index = (outer_idx * K0 + channel) * inner_size_ + inner_idx
      // bottom_index = (outer_idx * nthreads + index) * inner_size_ + inner_idx
      // nthreads = K0*group
    int top_offset = (nthreads/group)*inner_size;
    int bottom_offset = nthreads*inner_size;
    int channel = index % (nthreads/group);
    for (int i = 0; i < outer_num; ++i)
    {
      for (int j = 0; j < inner_size; ++j){
        weight_diff[index] += top_diff[channel*inner_size+j]*bottom_data[index*inner_size+j];
      }
      // shift batch
      top_diff += top_offset;
      bottom_data += bottom_offset;
    }
  }
}

template <typename Dtype>
void WeightedSumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const int outer_num = bottom[0]->count(0, axis_);
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const int count = this->blobs_[0]->count();
    // Gradient with respect to weight
    // sum_{out}(top_diff_patch.*bottom_data_patch)
    gpu_backward_weight<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom_data, inner_size_, outer_num, group_, weight_diff);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    const int outer_num = bottom[0]->count(0, axis_);
    const int count = this->blobs_[1]->count();
    // Gradient with respect to bias
    gpu_backward_bias<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, inner_size_, outer_num, bias_diff);
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const int outer_num = bottom[0]->count(0, axis_);
    const Dtype* weights = this->blobs_[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int offset = top[0]->count(axis_);
    // Gradient with respect to bottom data
    for (int i = 0; i < outer_num; ++i)
    {
      for (int k = 0; k < group_; ++k)
      {
        for (int j = 0; j < channel_out_; ++j)
        {
          caffe_gpu_axpby(inner_size_, weights[j], top_diff+j*inner_size_, Dtype(0), bottom_diff+j*inner_size_);
        }
        bottom_diff += channel_out_*inner_size_;
      }
      top_diff += offset;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightedSumLayer);

}  // namespace caffe
