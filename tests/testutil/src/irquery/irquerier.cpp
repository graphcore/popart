#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iterator>
#include <set>

#include <popart/logging.hpp>
#include <popart/tensorindex.hpp>

#include <testutil/irquery/irquerier.hpp>

namespace popart {
namespace testing {

Op *IrQuerier::graphHasOp(Graph &graph, OpPred opPred, Require testReq) {

  bool result   = false;
  Op *userValue = nullptr;

  auto &ops = graph.getOps();
  auto opIt = std::find_if(ops.begin(), ops.end(), [&opPred](auto &entry) {
    Op *op = entry.second.get();
    return opPred(op);
  });

  if (opIt != ops.end()) {
    userValue = opIt->second.get();
    result    = true;
  }

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected to find an op in " << graph.getGraphString()
       << " that matches predicate";

    triggerTestFailure(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect to find an op in " << graph.getGraphString()
       << " that matches predicate";

    triggerTestFailure(ss.str());
  }

  return userValue;
}

bool IrQuerier::opHasInputAt(Op *op, InIndex inIndex, Require testReq) {

  bool result = op->hasInput(inIndex);

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << op->str() << " to have input at index " << inIndex
       << " but input is not connected";

    triggerTestFailure(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << op->str() << " to have input at index "
       << inIndex << " ('" << op->inId(inIndex) << "')";

    triggerTestFailure(ss.str());
  }

  return result;
}

bool IrQuerier::opHasInputIdAt(Op *op,
                               InIndex inIndex,
                               TensorId inId,
                               Require testReq) {

  bool result = (op->hasInput(inIndex)) && (op->inId(inIndex) == inId);

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << op->str() << " to have input at index " << inIndex
       << " with ID '" << inId << "' ";
    if (!op->hasInput(inIndex)) {
      ss << "but input is not connected";
    } else {
      ss << "but got ID '" << op->inId(inIndex) << "'";
    }

    triggerTestFailure(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << op->str() << " to have input at index "
       << inIndex << " with ID '" << inId << "'";

    triggerTestFailure(ss.str());
  }

  return result;
}

bool IrQuerier::opHasInputIds(Op *op, TensorIds inIds, Require testReq) {

  std::vector<TensorId> actualVec;
  for (const auto &entry : op->input->tensorIdMap()) {
    actualVec.push_back(entry.second);
  }

  std::multiset<TensorId> expected(inIds.begin(), inIds.end());
  std::multiset<TensorId> actual(actualVec.begin(), actualVec.end());

  bool result = std::includes(
      actual.begin(), actual.end(), expected.begin(), expected.end());

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << op->str() << " to include inputs {"
       << logging::join(expected.begin(), expected.end(), ", ") << "} but got {"
       << logging::join(actualVec.begin(), actualVec.end(), ", ") << "}";

    triggerTestFailure(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << op->str() << "'s inputs to include {"
       << logging::join(expected.begin(), expected.end(), ", ") << "}";

    triggerTestFailure(ss.str());
  }

  return result;
}

bool IrQuerier::opHasExactInputIds(Op *op, TensorIds inIds, Require testReq) {

  std::vector<TensorId> actualVec;
  for (const auto &entry : op->input->tensorIdMap()) {
    actualVec.push_back(entry.second);
  }

  std::multiset<TensorId> expected(inIds.begin(), inIds.end());
  std::multiset<TensorId> actual(actualVec.begin(), actualVec.end());

  bool result = (actual == expected);

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << op->str() << " to have inputs {"
       << logging::join(expected.begin(), expected.end(), ", ") << "} but got {"
       << logging::join(actualVec.begin(), actualVec.end(), ", ") << "}";

    triggerTestFailure(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << op->str() << " to have inputs {"
       << logging::join(actualVec.begin(), actualVec.end(), ", ") << "}";

    triggerTestFailure(ss.str());
  }

  return result;
}

bool IrQuerier::opHasOutputAt(Op *op, OutIndex outIndex, Require testReq) {

  bool result = op->hasOutput(outIndex);

  if (testReq == Require::MustBeTrue && !result) {
    std::stringstream ss;
    ss << "Expected " << op->str() << " to have output at index " << outIndex
       << " but output is not connected";

    triggerTestFailure(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << op->str() << " to have output at index "
       << outIndex << " ('" << op->outId(outIndex) << "')";

    triggerTestFailure(ss.str());
  }

  return result;
}

bool IrQuerier::opHasOutputIdAt(Op *op,
                                OutIndex outIndex,
                                TensorId outId,
                                Require testReq) {

  bool result = (op->hasOutput(outIndex)) && (op->outId(outIndex) == outId);

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << op->str() << " to have output at index " << outIndex
       << " with ID '" << outId << "' ";
    if (!op->hasOutput(outIndex)) {
      ss << "but output is not connected";
    } else {
      ss << "but got ID '" << op->outId(outIndex) << "'";
    }

    triggerTestFailure(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << op->str() << " to have output at index "
       << outIndex << " with ID '" << outId << "'";

    triggerTestFailure(ss.str());
  }

  return result;
}

bool IrQuerier::opHasOutputIds(Op *op, TensorIds outIds, Require testReq) {

  std::vector<TensorId> actualVec;
  for (const auto &entry : op->output->tensorIdMap()) {
    actualVec.push_back(entry.second);
  }

  std::multiset<TensorId> expected(outIds.begin(), outIds.end());
  std::multiset<TensorId> actual(actualVec.begin(), actualVec.end());

  bool result = std::includes(
      actual.begin(), actual.end(), expected.begin(), expected.end());

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << op->str() << " to include outputs {"
       << logging::join(expected.begin(), expected.end(), ", ") << "} but got {"
       << logging::join(actualVec.begin(), actualVec.end(), ", ") << "}";

    triggerTestFailure(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << op->str() << "'s outputs to include {"
       << logging::join(expected.begin(), expected.end(), ", ") << "}";

    triggerTestFailure(ss.str());
  }

  return result;
}

bool IrQuerier::opHasExactOutputIds(Op *op, TensorIds outIds, Require testReq) {
  std::vector<TensorId> actualVec;
  for (const auto &entry : op->output->tensorIdMap()) {
    actualVec.push_back(entry.second);
  }

  std::multiset<TensorId> expected(outIds.begin(), outIds.end());
  std::multiset<TensorId> actual(actualVec.begin(), actualVec.end());

  bool result = (actual == expected);

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected " << op->str() << " to have outputs {"
       << logging::join(expected.begin(), expected.end(), ", ") << "} but got {"
       << logging::join(actualVec.begin(), actualVec.end(), ", ") << "}";

    triggerTestFailure(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect " << op->str() << " to have outputs {"
       << logging::join(actualVec.begin(), actualVec.end(), ", ") << "}";

    triggerTestFailure(ss.str());
  }

  return result;
}

void IrQuerier::triggerTestFailure(std::string errorMsg) {
  BOOST_REQUIRE_MESSAGE(false, errorMsg);
}

} // namespace testing
} // namespace popart
