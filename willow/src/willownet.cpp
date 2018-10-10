#include <fstream>
#include <iostream>
#include <sstream>
#include <willow/error.hpp>
#include <willow/filereader.hpp>
#include <willow/graph.hpp>
#include <willow/l1.hpp>
#include <willow/loss.hpp>
#include <willow/nll.hpp>
#include <willow/willownet.hpp>

namespace willow {

WillowNet::~WillowNet() = default;

WillowNet::WillowNet(std::string logDir_,
                     const EarlyInfo &earlyInfo,
                     const DataFlow &dataFlow,
                     const std::vector<Loss *> &losses) {

  // logDir_ is the path to the directory where all reading and writing
  // is done. here we just expand it to its canonical form
  // (not strictly needed)
  std::string canLogDir = io::getCanonicalDirName(logDir_);

  // note : must be the same as in driver.py:
  auto modelPath = io::appendDirFn(canLogDir, "model0.onnx");
  auto schedPath = io::appendDirFn(canLogDir, "schedule.txt");
  for (auto &x : std::vector<std::string>{modelPath, schedPath}) {
    io::confirmRegularFile(x);
  }

  // note : must be the same as in driver.py:
  std::string sectionHeader = ">>>>>>>>";
  auto input                = std::ifstream(schedPath, std::ios::in);
  std::map<std::string, std::vector<std::string>> sections;
  std::string globalFrag;
  std::string currentSection;
  while (std::getline(input, globalFrag)) {
    if (globalFrag.find(sectionHeader) != std::string::npos) {
      currentSection           = globalFrag.substr(sectionHeader.size() + 1);
      sections[currentSection] = {};
    } else {
      if (globalFrag.size() > 1) {
        sections[currentSection].push_back(globalFrag);
      }
    }
  }

  auto setVector = [&sections](std::string sectionName,
                               std::vector<TensorId> &v) {
    if (sections.find(sectionName) == sections.end()) {
      throw error("no section " + sectionName);
    }
    v = sections[sectionName];
  };

  auto setString = [&sections](std::string sectionName, std::string &s) {
    if (sections.find(sectionName) == sections.end()) {
      throw error("no section " + sectionName);
    }
    if (sections[sectionName].size() != 1) {
      throw error("cannot set single string for " + sectionName +
                  ", section of size " +
                  std::to_string(sections[sectionName].size()));
    }
    s = sections[sectionName][0];
  };

  std::string logdir;
  setString("log directory", logdir);

  std::vector<std::string> optimizerStrings;
  setVector("optimizer", optimizerStrings);

  std::vector<std::string> constTensors{};

  Optimizer optimizer{};
  auto model = io::getModel(modelPath);

  graph.reset(new Graph(std::move(model),
                        earlyInfo,
                        dataFlow,
                        losses,
                        std::move(optimizer),
                        std::move(constTensors),
                        logdir));

  std::stringstream ss2;
  graph->append(ss2);
  std::cout << ss2.str();
}

void WillowNet::connect(std::string backend) {}

void WillowNet::compile() {}

} // namespace willow
