// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/op.hpp>
#include <popart/op/averagepool.hpp>
#include <popart/op/globalaveragepool.hpp>
#include <popart/op/globalmaxpool.hpp>
#include <popart/op/maxpool.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popnn/Pooling.hpp>

namespace popart {
namespace popx {

static poplar::Type getReductionType(const popnn::PoolingType &pooling_type,
                                     const poplar::Type &input_type) {
  switch (pooling_type) {
  case popnn::PoolingType::AVG:
  case popnn::PoolingType::SUM:
    return poplar::FLOAT;
  case popnn::PoolingType::MAX:
    return input_type;
  }

  // TODO : Add stream operator for pooling_type
  throw error("Unknown pooling type");
}

class PoolOpx : public Opx {
public:
  PoolOpx(Op *op_, Devicex *device_) : Opx(op_, device_) {}

  popnn::pooling::PoolParams
  GetPoolingParameters(const popnn::PoolingType &pooling_type,
                       const TensorInfo &input_tensor,
                       const std::vector<int64_t> &kernel,
                       const std::vector<int64_t> &strides,
                       const std::vector<int64_t> &lowerPads,
                       const std::vector<int64_t> &upperPads) const {

    const auto &input_shape = input_tensor.shape_szt();

    const auto batch_size   = input_shape[0];
    const auto num_channels = input_shape[1];
    std::vector<std::size_t> input_field_shape(
        std::next(input_shape.begin(), 2), input_shape.end());

    std::vector<std::size_t> field_shape;
    std::vector<std::size_t> kernel_shape;
    std::vector<unsigned> stride;
    std::vector<int> padding_lower;
    std::vector<int> padding_upper;

    for (int d = 0; d < kernel.size(); d++) {
      field_shape.push_back(input_field_shape[d]);
      kernel_shape.push_back(kernel[d]);
      stride.push_back(static_cast<unsigned int>(strides[d]));
      padding_lower.push_back(static_cast<int>(lowerPads[d]));
      padding_upper.push_back(static_cast<int>(upperPads[d]));
    }

    auto data_type = getReductionType(pooling_type, popType(input_tensor));

    return {pooling_type,
            field_shape,
            kernel_shape,
            stride,
            padding_lower,
            padding_upper,
            num_channels,
            batch_size,
            data_type};
  }
};

// These templates use duck typing to expect that the methods are implement i.e.
// getSpatialK Consider if this could be improved with the use of an interface
// instead

template <class OP> class TPoolOpx : public PoolOpx {
public:
  TPoolOpx(Op *op_, Devicex *devicex_, const popnn::PoolingType pooling_type_)
      : PoolOpx(op_, devicex_), pooling_type(pooling_type_) {}

  void grow(poplar::program::Sequence &prog) const {

    OP &aOp = getOp<OP>();

    auto pool_params = GetPoolingParameters(pooling_type,
                                            op_p->inInfo(0),
                                            aOp.getSpatialK(),
                                            aOp.getStrides(),
                                            aOp.getLowerPads(),
                                            aOp.getUpperPads());

    logging::opx::debug(
        "Pooling InputField:{} Kernel:{} Strides:{} Pads L:{} U:{} C:{} Bs:{}",
        pool_params.inputFieldShape,
        pool_params.kernelShape,
        pool_params.stride,
        pool_params.inputTruncationOrPaddingLower,
        pool_params.inputTruncationOrPaddingUpper,
        pool_params.numChannels,
        pool_params.batchSize);

    setOutTensor(0,
                 popnn::pooling::pool(graph(),
                                      pool_params,
                                      getInTensor(0),
                                      prog,
                                      debugContext("pool"),
                                      dv_p->lowering().pooling_options));
  }

  popnn::PoolingType pooling_type;
};

template <class GRADOP, class OP> class TPoolGradOpx : public PoolOpx {
public:
  TPoolGradOpx(Op *op_,
               Devicex *devicex_,
               const popnn::PoolingType pooling_type_)
      : PoolOpx(op_, devicex_), pooling_type(pooling_type_) {}

  void grow(poplar::program::Sequence &prog) const {
    GRADOP &agOp = getOp<GRADOP>();

    auto pool_params = GetPoolingParameters(pooling_type,
                                            op_p->inInfo(0),
                                            agOp.creatorSpatialK,
                                            agOp.creatorStrides,
                                            agOp.creatorLowerPads,
                                            agOp.creatorUpperPads);

    logging::opx::debug("Pooling Grad InputField:{} Kernel:{} Strides:{} "
                        "Pads L:{} U:{} C:{} Bs:{}",
                        pool_params.inputFieldShape,
                        pool_params.kernelShape,
                        pool_params.stride,
                        pool_params.inputTruncationOrPaddingLower,
                        pool_params.inputTruncationOrPaddingUpper,
                        pool_params.numChannels,
                        pool_params.batchSize);

    setOutTensor(0,
                 popnn::pooling::poolInputGradient(
                     graph(),
                     pool_params,
                     getInTensor(GRADOP::getPrePooledInIndex()),
                     getInTensor(GRADOP::getPooledInIndex()),
                     getInTensor(GRADOP::getGradPooledInIndex()),
                     false, // useScaledVariant TODO T7295
                     prog,
                     debugContext("poolInputGradient"),
                     dv_p->lowering().pooling_options));
  }

  popnn::PoolingType pooling_type;
};

namespace {

OpxCreator<Opx>
    maxpoolOpxCreator({Onnx::Operators::MaxPool_1,
                       Onnx::Operators::MaxPool_8,
                       Onnx::Operators::MaxPool_10,
                       Onnx::Operators::MaxPool_11},
                      [](Op *op, Devicex *devicex) -> std::unique_ptr<Opx> {
                        return std::make_unique<TPoolOpx<MaxPoolOp>>(
                            op, devicex, popnn::PoolingType::MAX);
                      });

OpxCreator<Opx> maxpoolGradOpxCreator(
    {Onnx::GradOperators::MaxPoolGrad},
    [](Op *op, Devicex *devicex) -> std::unique_ptr<Opx> {
      return std::make_unique<TPoolGradOpx<MaxPoolGradOp, MaxPoolOp>>(
          op, devicex, popnn::PoolingType::MAX);
    });

OpxCreator<Opx> globalMaxpoolOpxCreator(
    {Onnx::Operators::GlobalMaxPool_1},
    [](Op *op, Devicex *devicex) -> std::unique_ptr<Opx> {
      return std::make_unique<TPoolOpx<GlobalMaxPoolOp>>(
          op, devicex, popnn::PoolingType::MAX);
    });

OpxCreator<Opx> globalMaxpoolGradOpxCreator(
    {Onnx::GradOperators::GlobalMaxPoolGrad},
    [](Op *op, Devicex *devicex) -> std::unique_ptr<Opx> {
      return std::make_unique<
          TPoolGradOpx<GlobalMaxPoolGradOp, GlobalMaxPoolOp>>(
          op, devicex, popnn::PoolingType::MAX);
    });

OpxCreator<Opx>
    averageOpxCreator({Onnx::Operators::AveragePool_1,
                       Onnx::Operators::AveragePool_7,
                       Onnx::Operators::AveragePool_10,
                       Onnx::Operators::AveragePool_11},
                      [](Op *op, Devicex *devicex) -> std::unique_ptr<Opx> {
                        return std::make_unique<TPoolOpx<AveragePoolOp>>(
                            op, devicex, popnn::PoolingType::AVG);
                      });

OpxCreator<Opx> averageGradOpxCreator(
    {Onnx::GradOperators::AveragePoolGrad},
    [](Op *op, Devicex *devicex) -> std::unique_ptr<Opx> {
      return std::make_unique<TPoolGradOpx<AveragePoolGradOp, AveragePoolOp>>(
          op, devicex, popnn::PoolingType::AVG);
    });

OpxCreator<Opx> globalAverageOpxCreator(
    {Onnx::Operators::GlobalAveragePool_1},
    [](Op *op, Devicex *devicex) -> std::unique_ptr<Opx> {
      return std::make_unique<TPoolOpx<GlobalAveragePoolOp>>(
          op, devicex, popnn::PoolingType::AVG);
    });

OpxCreator<Opx> globalAverageGradOpxCreator(
    {Onnx::GradOperators::GlobalAveragePoolGrad},
    [](Op *op, Devicex *devicex) -> std::unique_ptr<Opx> {
      return std::make_unique<
          TPoolGradOpx<GlobalAveragePoolGradOp, GlobalAveragePoolOp>>(
          op, devicex, popnn::PoolingType::AVG);
    });

} // namespace

} // namespace popx
} // namespace popart
