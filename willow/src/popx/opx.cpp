#include <poponnx/error.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/opx.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {
namespace popx {

Opx::Opx(Op *op_p_, Devicex *dv_p_) : op_p(op_p_), dv_p(dv_p_) {}

Opx::~Opx() = default;

poplar::Tensor Opx::createInput(int) const {
  throw error("Opx for " + op_p->op_type() + " cannot create Input");
}

bool Opx::createsEquiv(int, Opx *, int) const {
  throw error("No check for equivalent tensor create for type " +
              op_p->op_type());
}

std::vector<TensorId> Opx::mustExistBeforeCreate(int index0) const {
  throw error(
      "Opx for " + op_p->op_type() +
      " cannot say which poplar Tensors must exist to create at index " +
      std::to_string(index0));
}

void Opx::grow(poplar::program::Sequence &) const {
  throw error("adding poplar::Tensors not implemented for " + op_p->op_type());
}

bool Opx::canCreateInput(int) const { return false; }

poplar::Graph &Opx::graph() const { return dv_p->graph(); }

const poplar::Tensor &Opx::get(TensorId id) const {
  return dv_p->tensors.get(id);
}

void Opx::insert(TensorId id, const poplar::Tensor &tensor) const {
  dv_p->tensors.insert(id, tensor);
}

TensorId Opx::inId(InIndex index) const { return op_p->input->id(index); }
TensorId Opx::outId(OutIndex index) const { return op_p->output->id(index); }

Tensor *Opx::inTensor(InIndex index) const {
  return op_p->input->tensor(index);
}
Tensor *Opx::outTensor(OutIndex index) const {
  return op_p->output->tensor(index);
}

const TensorInfo &Opx::inInfo(InIndex index) const {
  return inTensor(index)->info;
}

const Shape &Opx::inShape(InIndex index) const { return inInfo(index).shape(); }

const TensorInfo &Opx::outInfo(OutIndex index) const {
  return outTensor(index)->info;
}

const Shape &Opx::outShape(OutIndex index) const {
  return outInfo(index).shape();
}

std::string Opx::idStr() const { return std::to_string(op_p->id); }

poplar::Tensor Opx::cloneNcopy(poplar::program::Sequence &prog,
                               TensorId id) const {
  auto outTensor = graph().clone(get(id));
  poplar::program::Copy copyProg(get(id), outTensor);
  prog.add(copyProg);
  return outTensor;
}

poplar::Tensor Opx::broadcast(const std::vector<int64_t> &desired_shape,
                              TensorId id) const {
  return broadcast(desired_shape, get(id));
}

poplar::Tensor Opx::broadcast(const std::vector<int64_t> &desired_shape,
                              poplar::Tensor t) const {
  const auto &t_shape = t.shape();

  // `new_shape` is `t_shape` prepended with ones to have matching rank as
  // `desired_shape`
  std::vector<std::size_t> new_shape(desired_shape.size(), 1);
  std::copy(t_shape.rbegin(), t_shape.rend(), new_shape.rbegin());

  // `t` now has matching rank as `desired_shape`
  t = t.reshape(new_shape);

  // Iteratively broadcast each mismatched dimension of `t`. This will
  // result in the shape of `t` matching `desired_shape`.
  for (int dim = 0; dim < desired_shape.size(); ++dim) {
    if (new_shape[dim] != desired_shape[dim] && new_shape[dim] != 1) {
      // Incompatible dimension found. Throw an exception, borrowing the same
      // terminology as numpy.
      throw error("np broadcasting failed, frames are not aligned");
    }

    if (new_shape[dim] != desired_shape[dim]) {
      t = t.broadcast(static_cast<unsigned>(desired_shape[dim]), dim);
    }
  }

  return t;
}

} // namespace popx
} // namespace poponnx
