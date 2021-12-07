// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <filereader.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <popart/dotvisualizer.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/call.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/topocons.hpp>

namespace popart {

DotVisualizer::DotVisualizer(std::string _check_) : check(_check_) {
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
  case TensorType::Unknown:
  case TensorType::ActGrad:
  case TensorType::N:
  default:
    return "\"black\"";
  }
}

std::ofstream &DotVisualizer::strm(const std::string &gString, const Ir &ir) {

  std::string abridgedGraphId;
  if (ir.getSessionOptions().separateCallOpPdfs == true) {
    abridgedGraphId = getAbridgedGraphName(gString);
  } else {
    abridgedGraphId = "all";
  }

  auto found = ofstreams.find(abridgedGraphId);
  if (found == ofstreams.end()) {
    // the full path to the .dot file to be written
    std::string dotfn = io::appendDirFn(ir.getSessionOptions().logDir,
                                        check + '_' + abridgedGraphId + ".dot");
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
  if (n->output->n()) {
    for (auto &x : n->input->tensorMap()) {
      auto regions = n->aliases(x.first, 0);
      inplace |=
          !std::all_of(regions.begin(),
                       regions.end(),
                       [](const view::Region &r) { return r.isEmpty(); });
    }
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
  if (tensorsVisited.count(tensor->id) == 0 || tensor->isGraphInput() ||
      tensor->isGraphOutput()) {
    tensorsVisited.insert(tensor->id);
    ofs << tensorDotId(tensor->id) << " [shape= \"egg\", label=\""
        << tensor->info << "  nc:" << tensor->consumers.getTotal()
        << (tensor->isGraphInput()
                ? "\ngraph input: " +
                      std::to_string(tensor->getGraphInputIndex())
                : "")
        << (tensor->isGraphOutput()
                ? "\ngraph output: " +
                      std::to_string(tensor->getGraphOutputIndex())
                : "")
        << "\", color = " << getTensorNodeColor(tensor->tensorType()) << "];\n";
  }
}

void DotVisualizer::write(const Ir &ir) {
  std::string upperCheck = check;
  // Cast to upper case
  std::transform(
      upperCheck.begin(), upperCheck.end(), upperCheck.begin(), ::toupper);

  if (!getDotChecks(ir).count("ALL") &&
      getDotChecks(ir).count(upperCheck) == 0) {
    return;
  }

  logging::ir::trace("Obtaining Op Schedule in DotVisualizer::write");
  auto scheduledOps = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);

  int start = std::max(0, ir.getSessionOptions().firstDotOp);
  int end   = std::min<int>(ir.getSessionOptions().finalDotOp,
                          static_cast<int>(scheduledOps.size()));

  if (!(start < end) && scheduledOps.size() != 0) {
    throw error("Invalid dot range ({}, {}) with schedule of size {}, "
                "as no Ops will be exported to the .dot file",
                ir.getSessionOptions().firstDotOp,
                ir.getSessionOptions().finalDotOp,
                scheduledOps.size());
  }

  for (int i = start; i < end; ++i) {
    auto &n = scheduledOps.at(i);

    auto generateNodeName = [this](Op *op) -> std::string {
      return nodeDotId(op->id);
    };

    // The string which will appear in the dot file to represent an Op
    std::stringstream coreNameStream;
    // add a graph identifier
    auto gString        = n->getGraph().id.str();
    bool addGraphPrefix = false;
    if (addGraphPrefix) {
      coreNameStream << '<' << getAbridgedGraphName(gString) << '>' << ' ';
    }
    coreNameStream << getNextGraphIndex(gString) << '.' << ' ' << n->opid.type
                   << " (" << n->id << ')';
    for (auto calledGraphId : n->getCalledGraphIds()) {
      coreNameStream << "<" << getAbridgedGraphName(calledGraphId.str()) << ">";
    }
    // Add the debug name if present and requested
    if (ir.getSessionOptions().dotOpNames) {
      if (!n->name().empty()) {
        coreNameStream << " (" << n->id << ":" << n->name() << ")";
      } else {
        coreNameStream << " (" << n->id << ")";
      }
    }
    if (n->hasVirtualGraphId()) {
      coreNameStream << " vgid:" << n->getVirtualGraphId();
    }

    if (n->hasExecutionPhase()) {
      coreNameStream << " pp:" << n->getExecutionPhase();
    }

    if (auto ipuCopy = dynamic_cast<IpuCopyOp *>(n)) {
      std::set<size_t> sIpus;
      for (auto ipu : ipuCopy->getSourceIpus()) {
        sIpus.insert(ipu.second);
      }
      if (sIpus.size() > 1) {
        coreNameStream << " sIpus: (";
        coreNameStream << logging::join(sIpus.begin(), sIpus.end(), ", ");
        coreNameStream << ")";
      } else {
        coreNameStream << " sIpu:" << ipuCopy->getSourceIpu();
      }
      coreNameStream << " dIpu:" << ipuCopy->getDestIpu();
    }
    coreNameStream << " ts:" << n->settings.tileSet;

    strm(gString, ir) << nodeDotId(n->id) << " [shape= \"box\", label=\""
                      << coreNameStream.str();
    strm(gString, ir) << "\", color = " << getOpNodeColor(n) << "];\n";

    for (auto &ind_ten : n->input->tensorMap()) {
      if (ind_ten.second->hasProducer() == false) {
        if (tensorsVisited.count(ind_ten.second->id) == 0) {
          makeNodeIfRequired(ind_ten.second, strm(gString, ir));
          for (auto &c : ind_ten.second->consumers.getOps()) {
            strm(gString, ir)
                << tensorDotId(ind_ten.second->id) << " -> "
                << generateNodeName(c) << " [color=black, label=\""
                << ind_ten.second->id << "\"];\n";
          }
        }
      }
    }

    for (auto &ind_ten : n->output->tensorMap()) {
      auto &consumers = ind_ten.second->consumers;
      for (auto &c : consumers.getOps()) {
        strm(gString, ir) << generateNodeName(n) << " -> "
                          << generateNodeName(c) << " [color=black, label=\""
                          << ind_ten.second->id << "\\n"
                          << ind_ten.second->info.shape() << "\"];\n";
      }

      if (consumers.getOps().size() == 0) {
        // must be an output
        makeNodeIfRequired(ind_ten.second, strm(gString, ir));
        strm(gString, ir) << generateNodeName(n) << " -> "
                          << tensorDotId(ind_ten.second->id)
                          << " [color=black, label=\"" << ind_ten.second->id
                          << "\"];\n";
      }
    }

    // For simplicity only show the after constraint
    for (auto &after : n->getGraph().topoCons->getAfters(n)) {
      strm(gString, ir) << generateNodeName(n) << " -> "
                        << generateNodeName(after)
                        << " [color=grey, style=dotted];\n";
    }
  }
  for (auto &x : ofstreams) {
    x.second << '}' << '\n';
    x.second.flush();
  }

  logging::ir::trace("Dot file(s) written");
}

std::set<std::string> DotVisualizer::getDotChecks(const Ir &ir) {
  // Cast to vector in order to later cast the values to upper
  std::vector<std::string> dotChecks{ir.getSessionOptions().dotChecks.begin(),
                                     ir.getSessionOptions().dotChecks.end()};

  auto popartDotChecks = getPopartEnvVar("DOT_CHECKS");

  if (popartDotChecks && (*popartDotChecks != "")) {
    std::vector<std::string> dotCheckFromPopartEnv;
    boost::split(dotCheckFromPopartEnv, *popartDotChecks, [](char c) {
      return c == ':';
    });

    for (auto &s : dotCheckFromPopartEnv) {
      dotChecks.push_back(s);
    }
  }

  // Cast to upper case
  for (auto &dotCheck : dotChecks) {
    std::transform(
        dotCheck.begin(), dotCheck.end(), dotCheck.begin(), ::toupper);
  }

  return std::set<std::string>{dotChecks.begin(), dotChecks.end()};
}

} // namespace popart
