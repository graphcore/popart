// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <testutil/irquery/tensorindexmaptestwrapper.hpp>

#include <testutil/irquery/testfailuretriggerer.hpp>

namespace popart {
namespace irquery {

namespace {
template <typename TensorIdContainer>
void outputIdList(std::ostream &out, const TensorIdContainer &ids) {
  bool isFirst = true;
  for (auto &id : ids) {
    if (!isFirst) {
      out << ", ";
    }
    out << "'" << id << "'";
    isFirst = false;
  }
}
} // namespace

TensorIndexMapTestWrapper::TensorIndexMapTestWrapper(
    Ir &ir_,
    const std::map<int, Tensor *> &tensorIndexMap_,
    const std::string &srcObjDescr_,
    const std::string &mapTypeDescrSingular_,
    const std::string &mapTypeDescrPlural_)
    : TestWrapper<std::map<int, Tensor *>>{ir_, tensorIndexMap_},
      srcObjDescr{srcObjDescr_}, mapTypeDescrSingular{mapTypeDescrSingular_},
      mapTypeDescrPlural{mapTypeDescrPlural_} {}

nonstd::optional<TensorIndexTestWrapper>
TensorIndexMapTestWrapper::hasId(const TensorId &id, Require testReq) {
  bool result = false;
  std::pair<int, Tensor *> value;
  std::vector<TensorId> actualVec;
  for (const auto &entry : wrappedObj) {
    actualVec.push_back(entry.second->id);
    if (entry.second->id == id) {
      result = true;
      value  = entry;
    }
  }

  if (testReq == Require::MustBeTrue && !result) {
    std::stringstream ss;
    ss << "Expected " << srcObjDescr << " to have " << mapTypeDescrSingular
       << " with ID '" << id << "' at some index (got " << mapTypeDescrPlural
       << ": ";
    outputIdList(ss, actualVec);
    ss << ")";

    triggerer->trigger(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << srcObjDescr << " to have "
       << mapTypeDescrSingular << " with ID '" << id << "' at any index (got '"
       << id << "' at index " << value.first << ")";

    triggerer->trigger(ss.str());
  }

  if (result) {
    return TensorIndexTestWrapper(
        ir, value, srcObjDescr, mapTypeDescrSingular, mapTypeDescrPlural);
  } else {
    return nonstd::optional<TensorIndexTestWrapper>();
  }
}

nonstd::optional<TensorIndexTestWrapper>
TensorIndexMapTestWrapper::hasIndex(int index, Require testReq) {

  auto it     = wrappedObj.find(index);
  bool result = it != wrappedObj.end();

  if (testReq == Require::MustBeTrue && !result) {
    std::stringstream ss;
    ss << "Expected " << srcObjDescr << " to have " << mapTypeDescrSingular
       << " at index " << index << " but " << mapTypeDescrSingular
       << " is not connected";

    triggerer->trigger(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << srcObjDescr << " to have "
       << mapTypeDescrSingular << " at index " << index << " (got '"
       << it->second->id << "' at index " << index << ")";

    triggerer->trigger(ss.str());
  }

  if (result) {
    return TensorIndexTestWrapper(
        ir, *it, srcObjDescr, mapTypeDescrSingular, mapTypeDescrPlural);
  } else {
    return nonstd::optional<TensorIndexTestWrapper>();
  }
}

nonstd::optional<TensorIndexTestWrapper>
TensorIndexMapTestWrapper::hasIdAtIndex(int index,
                                        const TensorId &id,
                                        Require testReq) {

  auto it     = wrappedObj.find(index);
  bool result = (it != wrappedObj.end() && it->second->id == id);

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << srcObjDescr << " to have " << mapTypeDescrSingular
       << " at index " << index << " with ID '" << id << "' ";
    if (it == wrappedObj.end()) {
      ss << "(" << mapTypeDescrSingular << " is not connected)";
    } else {
      ss << "(got '" << it->second->id << "' at index " << index << ")";
    }

    triggerer->trigger(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << srcObjDescr << " to have "
       << mapTypeDescrSingular << " at index " << index << " with ID '" << id
       << "'";

    triggerer->trigger(ss.str());
  }

  if (result) {
    return TensorIndexTestWrapper(
        ir, *it, srcObjDescr, mapTypeDescrSingular, mapTypeDescrPlural);
  } else {
    return nonstd::optional<TensorIndexTestWrapper>();
  }
}

bool TensorIndexMapTestWrapper::containsIds(const std::vector<TensorId> &ids,
                                            Require testReq) {
  std::vector<TensorId> actualVec;
  for (const auto &entry : wrappedObj) {
    actualVec.push_back(entry.second->id);
  }

  std::multiset<TensorId> expected(ids.begin(), ids.end());
  std::multiset<TensorId> actual(actualVec.begin(), actualVec.end());

  bool result = std::includes(
      actual.begin(), actual.end(), expected.begin(), expected.end());

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << srcObjDescr << "'s " << mapTypeDescrPlural
       << " to include {";
    outputIdList(ss, ids);
    ss << "} but got {";
    outputIdList(ss, actualVec);
    ss << "}";

    triggerer->trigger(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << srcObjDescr << "'s " << mapTypeDescrPlural
       << " to include {";
    outputIdList(ss, ids);
    ss << "}";

    triggerer->trigger(ss.str());
  }

  return result;
}

bool TensorIndexMapTestWrapper::hasExactIds(const std::vector<TensorId> &ids,
                                            Require testReq) {
  std::vector<TensorId> actualVec;
  for (const auto &entry : wrappedObj) {
    actualVec.push_back(entry.second->id);
  }

  std::multiset<TensorId> expected(ids.begin(), ids.end());
  std::multiset<TensorId> actual(actualVec.begin(), actualVec.end());

  bool result = (actual == expected);

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << srcObjDescr << "'s " << mapTypeDescrPlural
       << " to be {";
    outputIdList(ss, ids);
    ss << "} but got {";
    outputIdList(ss, actualVec);
    ss << "}";

    triggerer->trigger(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << srcObjDescr << "'s " << mapTypeDescrPlural
       << " to be {";
    outputIdList(ss, ids);
    ss << "}";

    triggerer->trigger(ss.str());
  }

  return result;
}

} // namespace irquery
} // namespace popart
