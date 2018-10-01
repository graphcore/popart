#include <neuralnet/pbwrap.hpp>

namespace neuralnet {

InputVecWrapper::InputVecWrapper(const std::vector<TensorId> &inputs_)
    : inputs(inputs_) {}
int InputVecWrapper::input_size() const {
  return static_cast<int>(inputs.size());
}

const TensorId &InputVecWrapper::input(int inIndex) const {
  return inputs.at(inIndex);
}

OutputVecWrapper::OutputVecWrapper(const std::vector<TensorId> &outputs_)
    : outputs(outputs_) {}
int OutputVecWrapper::output_size() const {
  return static_cast<int>(outputs.size());
}

const TensorId &OutputVecWrapper::output(int inIndex) const {
  return outputs.at(inIndex);
}

IOMapWrapper::IOMapWrapper(const std::map<int, TensorId> &M_) : M(M_) {
  max_index = 0;
  for (auto &index_id : M_) {
    max_index = std::max(max_index, index_id.first);
  }
}

const TensorId &IOMapWrapper::idAt(int inIndex) const {
  auto found = M.find(inIndex);
  if (found != M.end()) {
    return found->second;
  } else {
    return nullString;
  }
}

int IOMapWrapper::get_size() const { return max_index + 1; }

InputMapWrapper::InputMapWrapper(const std::map<int, TensorId> &M_)
    : IOMapWrapper(M_) {}

const TensorId &InputMapWrapper::input(int inIndex) const {
  return idAt(inIndex);
}

int InputMapWrapper::input_size() const { return get_size(); }

OutputMapWrapper::OutputMapWrapper(const std::map<int, TensorId> &M_)
    : IOMapWrapper(M_) {}

const TensorId &OutputMapWrapper::output(int inIndex) const { return idAt(inIndex); }

int OutputMapWrapper::output_size() const { return get_size(); }

} // namespace neuralnet
