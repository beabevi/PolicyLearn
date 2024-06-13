#include <torch/extension.h>

#include <vector>

torch::Tensor vrange_coalesced_cuda(
  torch::Tensor starts,
  torch::Tensor slices,
  torch::Tensor output,
  bool is_vrange
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &vrange_coalesced_cuda, "vrange forward (CUDA)");
}
