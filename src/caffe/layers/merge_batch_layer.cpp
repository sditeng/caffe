#include <vector>
#include <cfloat>
#include <algorithm>

#include "caffe/layer.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MergeBatchLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    ignore_blob_ = this->layer_param_.merge_batch_param().ignore_blob();
    int start_data_blob = ignore_blob_ ? 1 : 0;
    num_batches_ = bottom[start_data_blob]->shape(0);
    num_data_blob_ = bottom.size() - start_data_blob;
    if (ignore_blob_){
      CHECK_GT(bottom.size(), 2) <<
        " when ignore_blob is true, must contain at least two other blobs. ";
      CHECK_EQ(bottom[0]->shape(2), num_data_blob_) <<
        "ignore_blob must have shape 1 x 1 x DataBlobNumber x BatchNumber";
        CHECK_EQ(bottom[0]->shape(3), num_batches_) <<
        "ignore_blob must have shape 1 x 1 x DataBlobNumber x BatchNumber";
    }
    CHECK_EQ(top.size(), 1) << "Sum creates only one top blob. ";
    op_ = this->layer_param_.merge_batch_param().operation();
  }

template <typename Dtype>
void MergeBatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int start_data_blob = ignore_blob_ ? 1 : 0;
  num_batches_ = bottom[start_data_blob]->shape(0);
  for (int i = start_data_blob; i < bottom.size(); ++i)
  {
    CHECK(bottom[i]->shape() == bottom[start_data_blob]->shape()) << " Data blobs must have same size of channels.";
  }
  top[0]->ReshapeLike(*bottom[start_data_blob]);
}

template <typename Dtype>
void MergeBatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  int start_data_blob = ignore_blob_ ? 1 : 0;
  int inner_size = bottom[start_data_blob]->count(1);
  int count = top[0]->count();

  // set batch_coeff from bottom[0] if given
  const Dtype* batch_coeff = NULL;
  if (ignore_blob_){
     batch_coeff = bottom[0]->cpu_data();
  }
  // LOG(INFO) << batch_coeff.count() << ", " << caffe_cpu_asum(batch_coeff.count(), batch_coeff);
  switch (op_) {
    case MergeBatchParameter_MergeOp_SUM:
      // LOG(INFO) << inner_size;
      caffe_set(count, Dtype(0.0), top_data);
      for (int i = start_data_blob; i < bottom.size(); ++i) {
        const Dtype* bottom_data_i = bottom[i]->cpu_data();
        if (ignore_blob_)
        {
          for (int j = 0; j < num_batches_; ++j) {
            caffe_axpy(inner_size,  batch_coeff[j],
            bottom_data_i+inner_size*j, top_data+inner_size*j);
          }
          batch_coeff += num_batches_;
        } else {
          caffe_axpy(count, Dtype(1.0), bottom_data_i, top_data);
        }
      }
    break;
    case MergeBatchParameter_MergeOp_MAX:
      // Initialize
      caffe_set(count, Dtype(-FLT_MAX), top_data);
      for (int i = start_data_blob; i < bottom.size(); ++i) {
        const Dtype* bottom_data_i = bottom[i]->cpu_data();
        for (int j = 0; j < num_batches_; ++j)
        {
          if (!ignore_blob_ || batch_coeff[j] != Dtype(0)) {
            for (int k=0; k < inner_size; ++k) {
              if (top_data[k+j*inner_size] < bottom_data_i[j*inner_size+k]) {
                top_data[k+j*inner_size] = bottom_data_i[k+j*inner_size];
              }
            }
          }
        }
        if (ignore_blob_) {
          batch_coeff += num_batches_;
        }
      }
    break;
    default:
      LOG(FATAL) << "Unknown merge operation.";
  }
}

template <typename Dtype>
void MergeBatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int start_data_blob = ignore_blob_ ? 1 : 0;
  const Dtype* top_diff=top[0]->cpu_diff();
  const Dtype* top_data=top[0]->cpu_data();
  int inner_size = bottom[start_data_blob]->count(1);
  int count = bottom[start_data_blob]->count();
  const Dtype* batch_coeff = NULL;
  if (ignore_blob_)
  {
    batch_coeff = bottom[0]->cpu_data();
  }
  for (int i=start_data_blob; i<bottom.size(); i++){
    if(propagate_down[i]) {
      Dtype* bottom_diff_i = bottom[i]->mutable_cpu_diff();
      const Dtype* bottom_data_i = bottom[i]->cpu_data();
      caffe_set(count, Dtype(0), bottom_diff_i);
      for (int j = 0; j < num_batches_; ++j)
      {
        const Dtype coeff = ignore_blob_?batch_coeff[(i-start_data_blob)*num_batches_+j]:Dtype(1.0);
        switch (op_) {
          case MergeBatchParameter_MergeOp_SUM:
            if ( coeff == Dtype(1.0)) {
              caffe_copy(inner_size, top_diff+inner_size*j, bottom_diff_i+inner_size*j);
            } else {
              caffe_cpu_scale(inner_size, coeff, top_diff+inner_size*j, bottom_diff_i+inner_size*j);
            }
            break;
          case MergeBatchParameter_MergeOp_MAX:
            // equivalent to MIL, put diff to the bottom blob that is equal to top_data
            for (int k = 0; k < inner_size; ++k) {
              if (top_data[j*inner_size+k] == bottom_data_i[j*inner_size+k])
                bottom_diff_i[j*inner_size + k] = top_diff[j*inner_size+k];
            }
            break;
            case MergeBatchParameter_MergeOp_MEAN:
            break;
          default:
            LOG(FATAL) << "Unknown merge operation.";
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MergeBatchLayer);
#endif

INSTANTIATE_CLASS(MergeBatchLayer);
REGISTER_LAYER_CLASS(MergeBatch);

}  // namespace caffe
