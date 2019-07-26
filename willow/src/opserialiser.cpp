#include <popart/op.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

OpSerialiser::OpSerialiser(const Op *op, std::stringstream &ss_)
    : OpSerialiserBase(), ss(ss_) {
  ss << '\n' << "Op ";
  if (!op->getName().empty()) {
    ss << '"' << op->getName() << "\", ";
  }
  ss << op->id << " of type " << op->opid << '\n';

  int max_id_length =
      std::max(op->input->maxIdLength(), op->output->maxIdLength());

  ss << tab << "inputs" << '\n';
  op->input->append(ss, tab + tab, max_id_length);

  ss << '\n' << tab << "outputs" << '\n';
  op->output->append(ss, tab + tab, max_id_length);

  ss << '\n' << tab << "attributes" << '\n';
}

void OpSerialiser::appendAttribute(const std::string &name, float value) {
  appendAttr(name, value);
}

void OpSerialiser::appendAttribute(const std::string &name, int64_t value) {
  appendAttr(name, value);
}

void OpSerialiser::appendAttribute(const std::string &name, uint32_t value) {
  appendAttr(name, value);
}

void OpSerialiser::appendAttribute(const std::string &name, uint64_t value) {
  appendAttr(name, value);
}

void OpSerialiser::appendAttribute(const std::string &name,
                                   const std::vector<int64_t> &value) {
  appendAttr(name, value);
}

void OpSerialiser::appendAttribute(const std::string &name,
                                   const std::string &value) {
  appendAttr(name, value);
}

void OpSerialiser::appendAttribute(const std::string &name,
                                   boost::optional<int64_t> value) {
  if (value) {
    appendAttr(name, *value);
  }
}

void OpSerialiser::appendAttribute(const std::string &name, bool value) {
  appendAttr(name, value ? "true" : "false");
}

void OpSerialiser::appendAttribute(const std::string &name,
                                   const Scope &scope) {
  appendAttr(name, scope);
}

template <typename T>
void OpSerialiser::appendAttr(const std::string &name, const T &value) {
  ss << tab << tab << name << ": " << value << "\n";
}

// Ignore forward ops
void OpSerialiser::appendForwardOp(const Op *) {}

OpEquivIdCreator::OpEquivIdCreator(const Op *op) {
  ss << op->opid.domain << "::" << op->opid.type << "::" << op->opid.version
     << sep;

  appendAttr(*op->input.get());
  appendAttr(*op->output.get());
}

void OpEquivIdCreator::appendAttribute(const std::string &, float value) {
  appendAttr(value);
}

void OpEquivIdCreator::appendAttribute(const std::string &, int64_t value) {
  appendAttr(value);
}

void OpEquivIdCreator::appendAttribute(const std::string &, uint32_t value) {
  appendAttr(value);
}

void OpEquivIdCreator::appendAttribute(const std::string &, uint64_t value) {
  appendAttr(value);
}

void OpEquivIdCreator::appendAttribute(const std::string &,
                                       const std::vector<int64_t> &value) {
  appendAttr(value);
}

void OpEquivIdCreator::appendAttribute(const std::string &,
                                       const std::string &value) {
  appendAttr(value);
}

void OpEquivIdCreator::appendAttribute(const std::string &,
                                       boost::optional<int64_t> value) {
  if (value) {
    appendAttr(*value);
  } else {
    // something should always be written for
    // `value` when creating an equivalence id
    appendAttr('?');
  }
}

void OpEquivIdCreator::appendAttribute(const std::string &, bool value) {
  appendAttr(value);
}

void OpEquivIdCreator::appendAttribute(const std::string &,
                                       const Scope &scope) {
  appendAttr(scope);
}

void OpEquivIdCreator::appendForwardOp(const Op *op) {
  op->appendAttributes(*this);
}

std::string OpEquivIdCreator::str() { return ss.str(); }

template <> void OpEquivIdCreator::appendAttr(const TensorIndexMap &tmap) {
  int i = 0;
  for (auto &idx_tensor : tmap.tensorMap()) {
    auto idx    = idx_tensor.first;
    auto tensor = idx_tensor.second;

    if (i > 0) {
      ss << ',';
    }

    ss << idx << tensor->info.data_type() << tensor->info.shape();

    i++;
  }
  ss << sep;
}

template <typename T> void OpEquivIdCreator::appendAttr(const T &value) {
  ss << value << sep;
}

} // namespace popart
