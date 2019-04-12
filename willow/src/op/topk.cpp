#include <poponnx/makeunique.hpp>
#include <poponnx/op/topk.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

TopKOp::TopKOp(const OperatorIdentifier &opid_,
               int64_t K_,
               int64_t axis_,
               const Op::Settings &settings)
    : BaseSortOp(opid_, axis_, settings), K(K_) {}

std::unique_ptr<Op> TopKOp::clone() const { return make_unique<TopKOp>(*this); }

void TopKOp::setup() {

  validateAxis();

  auto shape = inShape(getInIndex());
  if (shape.at(getAxis()) < K) {
    throw error("Cannot take top-{} on dim of size {}, invalid Op {}",
                K,
                getAxis(),
                str());
  }

  shape[getAxis()] = K;

  // TODO T8133 : how to manage this correctly, should be INT64
  outInfo(getIndicesOutIndex()) = TensorInfo(DataType::INT32, shape);
  outInfo(getValuesOutIndex()) =
      TensorInfo(inInfo(getInIndex()).dataType(), shape);
}

void TopKOp::appendAttributes(OpSerialiserBase &os) const {
  BaseSortOp::appendAttributes(os);
  os.appendAttribute("K", K);
}

int64_t TopKOp::getK() const { return K; }

namespace {
std::unique_ptr<Op> topKFactory(const OperatorIdentifier &_opid,
                                const Op::Settings &settings,
                                const Attributes &attr) {
  // K is required
  int64_t K = attr.getAttribute<Attributes::Int>("k", 1);
  // axis is optional
  int64_t axis = attr.getAttribute<Attributes::Int>("axis", 0);

  return make_unique<TopKOp>(_opid, K, axis, settings);
}

static OpCreator<TopKOp>
    TopKOpCreator(Onnx::Operators::TopK_1, topKFactory, true);
} // namespace

} // namespace poponnx
