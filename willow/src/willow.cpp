#include <fstream>
#include <iostream>
#include <willow/error.hpp>
#include <willow/filereader.hpp>
#include <willow/graph.hpp>
#include <willow/l1.hpp>
#include <willow/loss.hpp>
#include <willow/willow.hpp>
#include <willow/nll.hpp>
#include <sstream>

namespace willow {

Willow::~Willow() = default;

Willow::Willow(std::string logDir_) {

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

  PreRunKnowledge preRunKnowledge;

  std::vector<TensorId> inNames;
  std::vector<TensorId> outNames;
  std::vector<TensorId> anchorNames;

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

  setVector("input names", inNames);
  setVector("output names", outNames);
  setVector("anchor names", anchorNames);

  Recorder recorder{anchorNames};

  std::string logdir;
  setString("log directory", logdir);

  std::vector<TensorId> lossStrings;
  setVector("losses", lossStrings);
  std::vector<std::unique_ptr<Loss>> losses;
  for (auto lossLine : lossStrings) {
    auto found = lossLine.find(':');
    if (found == std::string::npos) {
      throw error("invalid loss string in driver");
    }
    auto lossName = lossLine.substr(0, found);
    lossLine      = lossLine.substr(found + 1);
    switch (lossMap().at(lossName)) {
    case eLoss::NLL: {
      losses.push_back(std::unique_ptr<Loss>(new NllLoss(lossLine)));
      break;
    };
    case eLoss::L1: {
      losses.push_back(std::unique_ptr<Loss>(new L1Loss(lossLine)));
      break;
    }
    }
  }

  std::vector<std::string> optimizerStrings;
  setVector("optimizer", optimizerStrings);
  std::vector<std::string> dataInfoStrings;
  setVector("data info", dataInfoStrings);
  for (auto &infoLine : dataInfoStrings) {
    std::istringstream iss(infoLine);
    std::vector<std::string> frags;
    std::string frag;
    while (iss >> frag) {
      frags.push_back(frag);
    }
    if (frags.size() != 3) {
      throw error("expected [name type shape]");
    }
    std::string tensorName = frags[0];
    TensorInfo info(frags[1], frags[2]);
    preRunKnowledge.addInfo(tensorName, info);
  }

  // add learning rate to pre-run knowldege
  preRunKnowledge.addInfo(getLearningRateId(), {TP::FLOAT, {}});

  std::vector<std::string> constTensors{};

  Optimizer optimizer{};

  auto model = io::getModel(modelPath);
  std::vector<std::unique_ptr<Regularizer>> regularizers;

  graph.reset(new Graph(std::move(model),
                        std::move(preRunKnowledge),
                        std::move(recorder),
                        std::move(losses),
                        std::move(regularizers),
                        std::move(optimizer),
                        std::move(constTensors),
                        logdir));

  std::stringstream ss2;
  graph->append(ss2);
  std::cout << ss2.str();
}

void Willow::connect(std::string backend) {}

void Willow::compile() {}

} // namespace willow
