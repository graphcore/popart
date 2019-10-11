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
void OpSerialiser::appendForwardOp(const Op *op) {
  ss << tab << tab << op->debugName() << " attributes\n";
}

OpJsonSerialiser::OpJsonSerialiser(const Op *op, std::stringstream &ss_)
    : OpSerialiserBase(), ss(ss_) {

  ss << "{";

  appendKeyValue("type", op->opid.type);
  appendKeyValue("version", op->opid.version);
  appendKeyValue("domain", op->opid.domain);
  appendKeyValue("name", op->getName());

  appendKeyValues("inputs",
                  [&]() {
                    auto i = op->input->tensorMap().size();
                    for (auto it = op->input->tensorMap().begin();
                         it != op->input->tensorMap().end();
                         it++) {
                      ss << "{";
                      appendKeyValue("name", it->second->id);

                      std::stringstream s;
                      s << it->second->info.shape();

                      appendKeyValue("shape", s.str());
                      appendKeyValue("type", it->second->info.data_type());
                      appendKeyValue("index", it->first, true);

                      ss << "}";

                      if (i-- > 1) {
                        ss << ",";
                      }
                    }
                  }

  );

  appendKeyValues("outputs", [&]() {
    auto i = op->output->tensorMap().size();
    for (auto it = op->output->tensorMap().begin();
         it != op->output->tensorMap().end();
         it++) {
      ss << "{";
      appendKeyValue("name", it->second->id);
      appendKeyValue("index", it->first, true);
      ss << "}";

      if (i-- > 1) {
        ss << ",";
      }
    }
  });

  appendKeyValueFn(
      "attributes",
      [&]() {
        attributesAppended = false;
        op->appendAttributes(*this);

        if (attributesAppended)
          ss.seekp(-1, std::ios_base::end); // remove last ','
      },
      true);

  ss << "}";
}

template <typename T>
void OpJsonSerialiser::appendKeyValue(const std::string key,
                                      T value,
                                      bool last) {
  ss << "\"" << key << "\""
     << ":"
     << "\"" << value << "\"";

  if (!last)
    ss << ",";
}

void OpJsonSerialiser::appendKeyValueFn(const std::string key,
                                        std::function<void()> func,
                                        bool last) {
  ss << "\"" << key << "\""
     << ":"
     << "{";

  func();

  ss << "}";

  if (!last)
    ss << ",";
}

void OpJsonSerialiser::appendKeyValues(const std::string key,
                                       std::function<void()> func,
                                       bool last) {
  ss << "\"" << key << "\""
     << ":"
     << "[";

  func();

  ss << "]";

  if (!last)
    ss << ",";
}

void OpJsonSerialiser::appendAttribute(const std::string &name, float value) {
  appendAttr(name, value);
}

void OpJsonSerialiser::appendAttribute(const std::string &name, int64_t value) {
  appendAttr(name, value);
}

void OpJsonSerialiser::appendAttribute(const std::string &name,
                                       uint32_t value) {
  appendAttr(name, value);
}

void OpJsonSerialiser::appendAttribute(const std::string &name,
                                       uint64_t value) {
  appendAttr(name, value);
}

void OpJsonSerialiser::appendAttribute(const std::string &name,
                                       const std::vector<int64_t> &value) {
  appendAttr(name, value);
}

void OpJsonSerialiser::appendAttribute(const std::string &name,
                                       const std::string &value) {
  appendAttr(name, value);
}

void OpJsonSerialiser::appendAttribute(const std::string &name,
                                       boost::optional<int64_t> value) {
  if (value) {
    appendAttr(name, *value);
  }
}

void OpJsonSerialiser::appendAttribute(const std::string &name, bool value) {
  appendAttr(name, value ? "true" : "false");
}

void OpJsonSerialiser::appendAttribute(const std::string &name,
                                       const Scope &scope) {
  appendAttr(name, scope);
}

template <typename T>
void OpJsonSerialiser::appendAttr(const std::string &name, const T &value) {
  attributesAppended = true;
  appendKeyValue(name, value);
}

// Ignore forward ops
void OpJsonSerialiser::appendForwardOp(const Op *op) { (void)op; }

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
