#include <vector>

#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
  void WeightedSumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    bias_term_ = this->layer_param_.weighted_sum_param().bias_term();
    group_ = this->layer_param_.weighted_sum_param().group();
    axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.weighted_sum_param().axis());
    channel_in_ = bottom[0]->shape(axis_);
    CHECK_EQ(channel_in_ % group_, 0);
    channel_out_ = channel_in_/group_;
  // Dimensions starting from "axis_" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis_ == 1, N scale for a vector with dimension CHW are performed.
    inner_size_ = bottom[0]->count(axis_+1);
  // Check if we need to set up the weights
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      if (bias_term_) {
        this->blobs_.resize(2);
      } else {
        this->blobs_.resize(1);
      }
    // Intialize the weight
      vector<int> weight_shape(1);
      weight_shape[0] = channel_in_;
      this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.weighted_sum_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
      if (bias_term_) {
        vector<int> bias_shape(1, channel_out_);
        this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.weighted_sum_param().bias_filler()));
        bias_filler->Fill(this->blobs_[1].get());
      }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void WeightedSumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int channel_in = bottom[0]->shape(axis_);
  CHECK_EQ(channel_in_, channel_in)
  << "Input size incompatible with inner product parameters.";
  CHECK_EQ(channel_in_, channel_out_*group_);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis_ with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape[axis_] = channel_out_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void WeightedSumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias = NULL;
  if (bias_term_) {
   bias = this->blobs_[1]->cpu_data();
 }
  // The first "axis_" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
 const int outer_num = bottom[0]->count(0, axis_);
 const int offset = top[0]->count(axis_);
 caffe_set(outer_num*channel_out_*inner_size_, Dtype(0), top_data);
 for (int i = 0; i < outer_num; ++i)
 {
    // a patch of bottom_data inner_size_ = H*W
  for (int j = 0; j < channel_in_; ++j)
  {
    const int channel = j % channel_out_;
    caffe_axpy(inner_size_, weight[j], bottom_data, top_data+channel*inner_size_);
      bottom_data += inner_size_; //shift pointer
    }
    if (bias_term_) {
      for (int j = 0; j < channel_out_; ++j)
      {
        caffe_add_scalar(inner_size_, bias[j], top_data+j*inner_size_);
      }
    }
    top_data += offset;
  }
}

template <typename Dtype>
void WeightedSumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const int outer_num = bottom[0]->count(0, axis_);
    const int offset = top[0]->count(axis_);
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    // Gradient with respect to weight
    // sum(top_diff_patch.*bottom_data_patch)
    for (int i = 0; i < outer_num; ++i)
    {
      for (int j = 0; j < channel_in_; ++j)
      {
        const int channel = j % channel_out_;
        weight_diff[j] += caffe_cpu_dot(inner_size_, top_diff+channel*inner_size_, bottom_data);
        bottom_data += inner_size_;
      }
      top_diff += offset;
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const int outer_num = bottom[0]->count(0, axis_);
    Dtype bias_multiplier[inner_size_];
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    const int offset = top[0]->count(axis_);
    caffe_set(inner_size_, Dtype(1), bias_multiplier);
    // Gradient with respect to bias
    for (int i = 0; i < outer_num; ++i)
    {
      for (int j = 0; j < channel_out_; ++j)
      {
        bias_diff[j] += caffe_cpu_dot(inner_size_, top_diff+j*inner_size_, bias_multiplier);
      }
      top_diff += offset;
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const int outer_num = bottom[0]->count(0, axis_);
    const Dtype* weights = this->blobs_[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int offset = top[0]->count(axis_);
    // Gradient with respect to bottom data
    for (int i = 0; i < outer_num; ++i)
    {
      for (int j = 0; j < channel_in_; ++j)
      {
        const int channel = j % channel_out_;
        caffe_cpu_axpby(inner_size_, weights[j], top_diff+channel*inner_size_, Dtype(0), bottom_diff);
        bottom_diff += inner_size_;
      }
      top_diff += offset;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedSumLayer);
#endif

INSTANTIATE_CLASS(WeightedSumLayer);
REGISTER_LAYER_CLASS(WeightedSum);

}  // namespace caffe
