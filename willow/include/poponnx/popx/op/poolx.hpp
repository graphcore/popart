#ifndef GUARD_NEURALNET_POOLX_HPP
#define GUARD_NEURALNET_POOLX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

#include <popnn/Pooling.hpp>

namespace poponnx {

namespace popx {

class PoolOpx : public Opx {
public:
  PoolOpx(Op *, Devicex *);

  popnn::pooling::PoolParams
  GetPoolingParameters(const popnn::PoolingType &pooling_type,
                       const TensorInfo &input_tensor,
                       const std::vector<int64_t> &kernel,
                       const std::vector<int64_t> &strides,
                       const std::vector<int64_t> &lowerPads,
                       const std::vector<int64_t> &upperPads) const;
};

} // namespace popx
} // namespace poponnx

#endif
