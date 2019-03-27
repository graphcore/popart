#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/subgraph.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/topocons.hpp>

#include <poponnx/popx/subgraphoutlinex.hpp>

#include <sstream>

namespace poponnx {

static void printOps(const std::vector<Op *> &ops, std::string indent = "") {
  int i = 0;
  for (auto &o : ops) {

    int64_t subgraphId = 0;
    if (o->opid == Onnx::CustomOperators::Subgraph) {
      SubgraphOp *sgop = dynamic_cast<SubgraphOp *>(o);
      subgraphId       = sgop->getSubgraphId();
    }

    std::stringstream inputss;
    int j = 0;
    for (auto &in : o->input->tensors()) {
      if (j++ > 0) {
        inputss << ", ";
      }

      inputss << in->str();
    }
    std::stringstream outputss;
    j = 0;
    for (auto &out : o->output->tensors()) {
      if (j++ > 0) {
        outputss << ", ";
      }

      outputss << out->str();
    }

    logging::devicex::debug("{} {} : {} k:{} i:{} [{}] o:{} [{}]",
                            indent,
                            i++,
                            o->str(),
                            subgraphId,
                            o->input->n(),
                            inputss.str(),
                            o->output->n(),
                            outputss.str());

    if (o->opid == Onnx::CustomOperators::Subgraph) {
      SubgraphOp *sgop = dynamic_cast<SubgraphOp *>(o);
      printOps(sgop->getOps(), indent + "   ");
    }
  }
}

SubgraphOutlinex::SubgraphOutlinex() {}

bool SubgraphOutlinex::canApplyMatch(const std::vector<Op *> &ops,
                                     fwtools::subgraph::Match &m) {

  for (auto &start : m.starts) {
    for (int i = 0; i < m.length; ++i) {
      int index = start + i;

      // Reject any match that has a op that does not support caching
      if (ops[index]->supportsCaching() == false) {
        logging::err("{} does not support caching", ops[index]->str());
        return false;
      }
    }
  }

  return true;
}

int64_t SubgraphOutlinex::getNextSubgraphId() {
  static int64_t counter = 1;
  return counter++;
}

std::vector<Op *> SubgraphOutlinex::getOutlineView(const std::vector<Op *> &ops,
                                                   const Ir &ir) {

  std::vector<Op *> outlinedOps = ops;

  logging::devicex::debug(
      "SubgraphOutlinex::getOutlineView Op list before outlining");
  printOps(outlinedOps);

  auto matches = fwtools::subgraph::getMatches(outlinedOps, 1.0f);

  // Sort the matches so we do the smallest subgraphs first
  std::sort(matches.begin(),
            matches.end(),
            [=](fwtools::subgraph::Match &p1, fwtools::subgraph::Match &p2) {
              return p1.length < p2.length;
            });

  logging::devicex::debug("Outline matches");
  for (auto &m : matches) {
    logging::devicex::debug("{}", m);
  }

  // For each match
  for (int matchIndex = 0; matchIndex < matches.size(); ++matchIndex) {
    auto &currentMatch = matches[matchIndex];

    // Check that we can apply this match, skip if we can not
    if (canApplyMatch(outlinedOps, currentMatch) == false) {
      logging::devicex::info("Not outline match ({})", currentMatch);
      continue;
    } else {
      logging::devicex::debug("Appying outline match ({})", currentMatch);
    }

    // Get a unique id for this outline
    int64_t subgraphId = getNextSubgraphId();

    // For occurange of a match starting at the end .
    for (auto it = currentMatch.starts.rbegin();
         it != currentMatch.starts.rend();
         ++it) {

      int inIndex  = 0;
      int outIndex = 0;

      // Create the new subgraphOp.
      // It is frustraing that we have to const cast the ir, but the devicex was
      // not suppose to change the ir but we need a new op id for the subgraph
      // op which requires const casting the ir
      //
      SubgraphOp *sgop = new SubgraphOp(const_cast<Ir &>(ir), subgraphId);
      subgraphOps.push_back(std::unique_ptr<Op>(sgop));

      // Push the subgraph op in t the list of ops at the start of the match,
      // later we will remove the ops in the match
      outlinedOps.insert(outlinedOps.begin() + *it, sgop);

      sgop->getChildOpsInfo().resize(currentMatch.length);

      // Work out which tensors are intenal to this subgraph
      std::vector<Tensor *>
          internalTensors; // List of tensors internal to the subgraph
      std::vector<Tensor *> outputTensors; // List of output tensors

      // For each op in the match occurance
      for (int k = 0; k < currentMatch.length; ++k) {
        Op *so = *(outlinedOps.begin() + *it + 1 + k);

        // For each output of the op in the subgraph
        for (auto &o : so->output->tensorMap()) {

          // Add the output to the subgraph output's
          outputTensors.push_back(o.second);

          // Get the consumers of the output
          auto consumers = o.second->consumers.getOps();

          // Remove consumer ops that are in the subgraph
          for (int j = 0; j < currentMatch.length; ++j) {
            Op *so2 = *(outlinedOps.begin() + *it + 1 + j);
            consumers.erase(
                std::remove(consumers.begin(), consumers.end(), so2),
                consumers.end());
          }

          // If all consumers of the output are removed then the output in not
          // external to the subgraph
          if (consumers.size() == 0) {

            // but it may be an anchor - in which case it still needs to be an
            // output of the subgraph
            if (std::find(ir.getDataFlow().anchors().begin(),
                          ir.getDataFlow().anchors().end(),
                          o.second->id) == ir.getDataFlow().anchors().end()) {
              // At the output tensor the list of internalTensors
              internalTensors.push_back(o.second);
            }
          }
        }
      }

      // For each op in the match occurance
      for (int k = 0; k < currentMatch.length; ++k) {
        Op *so = *(outlinedOps.begin() + *it + 1 + k);

        SubgraphOp::OpInfo &opi = sgop->getChildOpsInfo()[k];
        opi.op                  = so;

        // Get the op's intput and output tensor map
        auto inputMap  = so->input->tensorMap();
        auto outputMap = so->output->tensorMap();

        // Set the phase & virtualGraphId to the same as the first op in the
        // subgraph
        if (k == 0) {
          sgop->setPhase(so->getPhase());
          sgop->setVirtualGraphId(so->getVirtualGraphId());
        }

        // For each input to the op
        for (auto &t : inputMap) {

          // Is the input internal or external to the subgraph
          if (std::find(internalTensors.begin(),
                        internalTensors.end(),
                        t.second) == internalTensors.end()) {
            // External

            // And it is not an output
            if (std::find(outputTensors.begin(),
                          outputTensors.end(),
                          t.second) == outputTensors.end()) {
              sgop->connectInTensor(inIndex++, t.second->id);
              opi.inputs[t.first] = {t.second->id, true};
            } else {
              opi.inputs[t.first] = {t.second->id, false};
            }
          } else {
            // Internal
            opi.inputs[t.first] = {t.second->id, false};
          }
        }

        // For each output from the op
        for (auto &t : outputMap) {

          // Is the output internal or external to the subgraph
          if (std::find(internalTensors.begin(),
                        internalTensors.end(),
                        t.second) == internalTensors.end()) {
            // External
            sgop->connectOutTensor(outIndex++, t.second->id);
            opi.outputs[t.first] = {t.second->id, true};
          } else {
            // Internal
            opi.outputs[t.first] = {t.second->id, false};
          }
        }
      }

      // Remove the ops that have been outlined. The +1 is to account for the
      // subgraphOp we added
      outlinedOps.erase(outlinedOps.begin() + *it + 1,
                        outlinedOps.begin() + *it + 1 + currentMatch.length);
    }

    // We now need to adjust the remaining matches index to account for them
    // removal of the subgraph ops
    // 1. We will decrease index's smaller that the start of the each match by
    // the length of the match + 1
    // 2. We will reduce the lenghts of matches will are a supper set of the
    // current match.

    // For each of the remaining matches
    for (int nextMatchIndex = matchIndex + 1; nextMatchIndex < matches.size();
         ++nextMatchIndex) {
      auto &nextMatch    = matches[nextMatchIndex];
      auto origNextMatch = nextMatch;

      bool nextMatchChanged = false;

      // For each occurange of the next matches
      for (auto nextMatchStartIt = nextMatch.starts.begin();
           nextMatchStartIt != nextMatch.starts.end();
           ++nextMatchStartIt) {

        int nextMatchStartIndex = *nextMatchStartIt;
        int nextMatchLength     = nextMatch.length;

        // For each of the current matches
        for (auto currentMatchStartIt = currentMatch.starts.begin();
             currentMatchStartIt != currentMatch.starts.end();
             ++currentMatchStartIt) {

          int currentMatchStartIndex = *currentMatchStartIt;

          // If the start index of the next match is greater than the current
          // match then adjust the start index of the next match
          if (nextMatchStartIndex > currentMatchStartIndex) {
            *nextMatchStartIt = *nextMatchStartIt - currentMatch.length + 1;
            nextMatchChanged  = true;
          }

          // Reduce the length of the match if the current match is a subset of
          // the next match
          if (nextMatchStartIt == nextMatch.starts.begin()) {
            if ((currentMatchStartIndex >= nextMatchStartIndex) &&
                (currentMatchStartIndex + currentMatch.length <=
                 nextMatchStartIndex + nextMatchLength)) {
              nextMatch.length = nextMatch.length - currentMatch.length + 1;
              nextMatchChanged = true;
            }
          }
        }
      }

      if (nextMatchChanged) {
        logging::devicex::debug(
            "Match changed from {} to {}", origNextMatch, nextMatch);
      }
    }

    /*
    if (matchIndex + 1 < matches.size()) {
      logging::err("Rejusted remaining matches");
      for (int j = matchIndex + 1; j < matches.size(); ++j) {
        logging::devicex::debug("{}", matches[j]);
      }
    }
    */
  }

  logging::devicex::debug(
      "SubgraphOutlinex::getOutlineView Op list after outlining");
  printOps(outlinedOps);

  return outlinedOps;
}

} // namespace poponnx
