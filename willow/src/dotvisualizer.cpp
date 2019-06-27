#include <boost/algorithm/string.hpp>
#include <fstream>
#include <sstream>
#include <poponnx/dotvisualizer.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/call.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensornames.hpp>

namespace poponnx {

DotVisualizer::DotVisualizer(const Ir *_ir_, DotCheck _check_)
    : ir(_ir_), check(_check_) {
  // Register the root graph
  graphMapping.insert({"", "r"});
}

DotVisualizer::AbridgedGraphName
DotVisualizer::getAbridgedGraphName(const FullGraphName &gString) {
  auto foundG = graphMapping.find(gString);
  if (foundG == graphMapping.end()) {
    AbridgedGraphName aName = gString;
    // taking the digit in the braces (otherwise use full name)
    auto found_open  = gString.find('(');
    auto found_close = gString.find(')');
    if (found_open != std::string::npos && found_close != std::string::npos &&
        found_close > found_open) {
      aName = gString.substr(found_open + 1, found_close - found_open - 1);
    }
    graphMapping[gString] = aName;
    return graphMapping[gString];
  }
  return foundG->second;
}

std::string DotVisualizer::nodeDotId(OpId id) const {
  return "\"n_" + std::to_string(id) + "\"";
}

std::string DotVisualizer::tensorDotId(const TensorId &id) const {
  return '\"' + id + '\"';
}

std::string DotVisualizer::getTensorNodeColor(TensorType type) const {
  switch (type) {
  case TensorType::Stream:
    return "\"red\"";
  case TensorType::Const:
    return "\"gold\"";
  case TensorType::Variable:
    return "\"blue\"";
  case TensorType::Momentum:
  case TensorType::Unknown:
  case TensorType::ActGrad:
  case TensorType::N:
  default:
    return "\"black\"";
  }
}

std::ofstream &DotVisualizer::strm(const std::string &gString) {

  std::string abridgedGraphId;
  if (ir->getSessionOptions().separateCallOpPdfs == true) {
    abridgedGraphId = getAbridgedGraphName(gString);
  } else {
    abridgedGraphId = "all";
  }

  auto found = ofstreams.find(abridgedGraphId);
  if (found == ofstreams.end()) {
    // the full path to the .dot file to be written
    std::string dotfn = io::appendDirFn(ir->getSessionOptions().logDir,
                                        getDotCheckString(check) + '_' +
                                            abridgedGraphId + ".dot");
    logging::ir::info("Appending to open dot file {}", dotfn);
    ofstreams[abridgedGraphId] = std::ofstream{};
    auto &thisStream           = ofstreams[abridgedGraphId];
    thisStream.open(dotfn, std::ios::out);
    if (!thisStream.is_open()) {
      throw error("failed to open file `" + dotfn + '\'');
    }
    thisStream << "digraph net {\n";
    thisStream << "size=\"6,6\";\n";
    return thisStream;
  }
  return found->second;
}

std::string DotVisualizer::getOpNodeColor(Op *n) {
  bool inplace = false;
  for (auto &x : n->input->tensorMap()) {
    inplace |= !n->aliases(x.first).isEmpty();
  }

  if (inplace) {
    return "\"#6B8E23\"";
  }
  return "black";
}

int DotVisualizer::getNextGraphIndex(const FullGraphName &gString) {
  auto found = graphScheduleCounter.find(gString);
  if (found == graphScheduleCounter.end()) {
    graphScheduleCounter.insert({gString, 0});
    return 0;
  }
  ++found->second;
  return found->second;
}

// Create a tensor node in the .dot file
void DotVisualizer::makeNodeIfRequired(const Tensor *tensor,
                                       std::ofstream &ofs) {
  if (tensorsVisited.count(tensor->id) == 0) {
    tensorsVisited.insert(tensor->id);
    ofs << tensorDotId(tensor->id) << " [shape= \"egg\", label=\""
        << tensor->info << "  nc:" << tensor->consumers.getTotal()
        << "\", color = " << getTensorNodeColor(tensor->tensorType()) << "];\n";
  }
}

void DotVisualizer::write() {
  if (getDotChecks().count(check) == 0) {
    return;
  }

  logging::ir::trace("Obtaining Op Schedule");
  auto scheduledOps = ir->getOpSchedule({});

  int start = std::max(0, ir->getSessionOptions().firstDotOp);
  int end   = std::min<int>(ir->getSessionOptions().finalDotOp,
                          static_cast<int>(scheduledOps.size()));

  if (!(start < end) && scheduledOps.size() != 0) {
    throw error("Invalid dot range ({}, {}) with schedule of size {}, "
                "as no Ops will be exported to the .dot file",
                ir->getSessionOptions().firstDotOp,
                ir->getSessionOptions().finalDotOp,
                scheduledOps.size());
  }

  for (int i = start; i < end; ++i) {
    auto &n = scheduledOps.at(i);

    // The string which will appear in the dot file to represent an Op
    std::stringstream coreNameStream;
    // add a graph identifier
    auto gString        = n->getGraph().id.str();
    bool addGraphPrefix = false;
    if (addGraphPrefix) {
      coreNameStream << '<' << getAbridgedGraphName(gString) << '>' << ' ';
    }
    coreNameStream << getNextGraphIndex(gString) << '.' << ' ' << n->opid.type;
    if (dynamic_cast<CallOp *>(n)) {
      auto calledGraphId = dynamic_cast<CallOp *>(n)->getCalledGraph().id.str();
      coreNameStream << "<" << getAbridgedGraphName(calledGraphId) << ">";
    }
    // Add the debug name if present and requested
    if (ir->getSessionOptions().dotOpNames) {
      if (!n->name().empty()) {
        coreNameStream << " (" << n->name() << ")";
      } else {
        coreNameStream << " (" << n->id << ")";
      }
    }

    strm(gString) << nodeDotId(n->id) << " [shape= \"box\", label=\""
                  << coreNameStream.str();
    strm(gString) << "\", color = " << getOpNodeColor(n) << "];\n";

    // insert the input -> op edges into the .dot file
    for (auto &ind_ten : n->input->tensorMap()) {
      TensorId tenId = ind_ten.second->id;
      makeNodeIfRequired(ind_ten.second, strm(gString));
      strm(gString) << tensorDotId(tenId) << " -> " << nodeDotId(n->id)
                    << "[color=gray]" << ';' << '\n';
    }

    // insert the op -> output edges into the .dot file
    for (auto &ind_ten : n->output->tensorMap()) {
      auto tenId = ind_ten.second->id;
      makeNodeIfRequired(ind_ten.second, strm(gString));
      strm(gString) << nodeDotId(n->id) << " -> " << tensorDotId(tenId)
                    << "[color=gray]" << ';' << '\n';
      TensorId possibleGradId = getGradId(tenId);
    }
  }

  for (auto &x : ofstreams) {
    x.second << '}' << '\n';
    x.second.flush();
  }

  logging::ir::trace("Dot file(s) written");
}

std::set<DotCheck> DotVisualizer::getDotChecks() {
  auto dotChecks = ir->getSessionOptions().dotChecks;

  auto poponnxDotChecks = std::getenv("POPONNX_DOT_CHECKS");
  if (poponnxDotChecks && std::strcmp(poponnxDotChecks, "") != 0) {
    std::vector<std::string> dotCheckStrings;
    boost::split(
        dotCheckStrings, poponnxDotChecks, [](char c) { return c == ':'; });

    for (auto &s : dotCheckStrings) {
      auto c = dotCheckFromString(s);
      dotChecks.insert(c);
    }
  }

  return dotChecks;
}

} // namespace poponnx
