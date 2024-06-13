#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>

#define iceil(x, y) ((x / y) + (x % y != 0))

__global__
void vrange_uncoalesced_cuda_kernel(
  const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> starts,
  const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> slices,
  torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> output
) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int slice_end = slices[threadId];
  int out_start = threadId > 0 ? slices[threadId - 1] : 0;
  int length = slice_end - out_start;

  int start = starts[threadId];
  for (int i = 0; i < length; ++i) {
    output[out_start + i] = start + i;
  }
}


template <bool is_vrange>
__global__
void vrange_coalesced_cuda_kernel(
  const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> starts,
  const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> slices,
  torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> output,
  int input_size,
  int group_size
) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int group = threadId / group_size;
  int warpId = threadId % group_size;

  if (group < input_size) {
    int end = slices[group];
    int out_start = group > 0 ? slices[group - 1] : 0;
    int length = end - out_start;

    int it_needed = iceil(length, group_size);
    it_needed += 1;

    int start = starts[group];

    for (int i = 0; i < it_needed; ++i) {
      int offset = i * group_size + warpId;
      if (offset < length) {
        output[out_start + offset] = start + (is_vrange ? offset : 0);
      }
    }
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor vrange_coalesced_cuda(
  torch::Tensor starts,
  torch::Tensor slices,
  torch::Tensor output,
  bool is_vrange
) {
  CHECK_INPUT(starts);
  CHECK_INPUT(slices);
  TORCH_CHECK(starts.sizes().size() == 1, "starts doesn't have rank 1")
  TORCH_CHECK(slices.sizes().size() == 1, "slices doesn't have rank 1")
  TORCH_CHECK(output.sizes().size() == 1, "output doesn't have rank 1")

  int input_size = starts.size(0);
  auto output_size = slices[-1].item<int64_t>();

  TORCH_CHECK(output.size(0) == output_size, "output size smaller than end slice");

  auto group_size = iceil(output_size, input_size);

  auto num_threads = group_size * input_size;

  // auto options = torch::TensorOptions().dtype(torch::kInt64);
  // auto output = torch::zeros({output_size}, options);

  auto blocks = iceil(num_threads, 1024);

  if (is_vrange) {
    vrange_coalesced_cuda_kernel<true><<<blocks, 1024>>>(
      starts.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
      slices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
      output.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
      input_size,
      group_size
    );
  } else {
    vrange_coalesced_cuda_kernel<false><<<blocks, 1024>>>(
      starts.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
      slices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
      output.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
      input_size,
      group_size
    );
  }
  return output;
}
