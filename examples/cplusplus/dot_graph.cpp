// A tool for visualising the Ir (PopONNX's Intermediate representation)
// as a .dot file which can be compiled into a .pdf file.
// This tool supports training and as well as inference.

#include <iostream>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/graphtransformer.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/optionflags.hpp>

using namespace poponnx;

// https://stackoverflow.com/questions/4654636
namespace {
bool is_number(const std::string &s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}
} // namespace

// TODO: improved visualization (T5144)
// TODO: use boost program options (T7217)

int main(int argc, char **argv) {

  logging::ir::info("{} user command-line parameters received", argc - 1);

  int nArgsExpected = 6;
  if (argc - 1 != nArgsExpected) {
    std::stringstream ss;
    // The command-line parameters:
    ss << "\nExpected " << nArgsExpected << " command-line parameters:"
       << "\n\t (1) path of the ONNX model file"
       << "\n\t (2) path of the output directory"
       << "\n\t (3) compile dot->pdf(s) (0/1)"
       << "\n\t (4) train with NLL on output (0/1)"
       << "\n\t (5) the first op in the schedule to export will be min(0, (5))"
       << "\n\t (6) the final op in the schedule to export will be "
       << "max(n-ops in the schedule, (6))"
       << "\n\t Example usage : "
       << "./dot_inference_graph /path/to/model.onnx . 1 1 10 100 "
       << "\n\t will generate a pdf for model.onnx in the current directory, "
       << "all Ops 10->100.";

    // note for (3):
    // - unix only, uses std::system
    // - requires the dot program to be installed "./dot --help"
    std::cout << ss.str() << std::endl;
    return 0;
  }

  std::string modelPath = argv[1];
  logging::ir::info("modelPath set to {}", modelPath);
  std::string outputDir = argv[2];
  logging::ir::info("outputDir set to {}", modelPath);

  // compile to PDF?
  std::string argv3 = std::string(argv[3]);
  if (argv3 != "0" && argv3 != "1") {
    throw error("Expected arg 3 to be '0' or '1', not {}", argv3);
  }
  bool compileDotsToPDF = std::stoi(argv3);
  logging::ir::info("compileDotsToPDF set to {}", compileDotsToPDF);

  // train?
  std::string argv4 = std::string(argv[4]);
  if (argv4 != "0" && argv4 != "1") {
    throw error("Expected arg 4 to be '0' or '1', not {}", argv4);
  }
  bool trainWithNll = std::stoi(argv4);
  logging::ir::info("trainWithNll set to {}", trainWithNll);

  // the first op in the schedule to visualize
  std::string argv5 = std::string(argv[5]);
  if (!is_number(argv5)) {
    throw error("Expected arg 5 to be a number, not {}", argv5);
  }

  // the final op in the schedule to visualize
  std::string argv6 = std::string(argv[6]);
  if (!is_number(argv6)) {
    throw error("Expected arg 6 to be a number, not {}", argv6);
  }

  SessionOptions opts;
  opts.firstDotOp = std::stoi(argv5);
  opts.finalDotOp = std::stoi(argv6);

  GraphTransformer gt(modelPath);
  gt.convertAllFixedPointInitializersToConstants();

  // The ONNX Model might have been exported with inference in mind.
  // In general this makes no difference, but for BatchNormalization
  // (and potentially other Operator types) the number of outputs
  // is 5 for training and 1 for inference. This call appends dummy
  // output names to BatchNormalization (and possibly makes other adjustments)
  if (trainWithNll) {
    gt.prepareNodesForTraining();
  }
  auto modelProtoString = gt.getModelProto();
  auto modelProto       = io::getModelFromString(modelProtoString);

  std::vector<std::string> dotStrings;

  // Currently only the FINAL .dot file. Append others here as required.
  opts.dotChecks.insert(DotCheck::FINAL);

  opts.logDir = outputDir;

  if (modelProto.graph().output_size() == 0) {
    throw error("Cannot generate dot for ONNX graph with no output");
  }

  auto out      = modelProto.graph().output(0).name();
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  std::unique_ptr<Optimizer> optimizer;
  std::vector<std::unique_ptr<Loss>> up_losses;

  TensorId label = "label";
  InputShapeInfo isi;

  if (trainWithNll) {
    optimizer.reset(new ConstSGD(0.01f));

    if (modelProto.graph().output_size() != 1) {
      // requires extension for training with more than 1 output
      throw error("Cannot call dot_graph TRAIN, model only has one output");
    }

    // choosing an arbitrary shape for label, can't run shape inference now
    isi.add(label, {"INT32", std::vector<int64_t>{7}});

    up_losses.emplace_back(
        std::unique_ptr<Loss>(new NllLoss(out, label, "nllLossVal")));
  }

  std::vector<Loss *> losses;
  for (auto &x : up_losses) {
    losses.push_back(x.get());
  }

  Ir ir;
  ir.prepare({modelProto,
              isi,
              dataFlow,
              losses,
              optimizer.get(),
              opts,
              Patterns(PatternsLevel::DEFAULT).enableInPlace(true)});

  // verify that the dot files have been created
  auto dotFileNames =
      io::getMatchFns(io::getCanonicalDirName(opts.logDir), ".dot");
  if (dotFileNames.size() == 0) {
    throw error("Error in dot_inference_graph, no .dot files in {}",
                opts.logDir);
  }

  // dot -Tpdf -o x.pdf x.dot
  if (compileDotsToPDF == true) {
    for (auto check : opts.dotChecks) {
      auto dot_string = getDotCheckString(check);
      std::stringstream command_ss;
      command_ss << "dot "
                 << " -Tpdf "
                 << " -o " << io::appendDirFn(opts.logDir, dot_string + ".pdf")
                 << " " << io::appendDirFn(opts.logDir, dot_string + ".dot");
      std::string command = command_ss.str();
      int ran             = std::system(command.c_str());
      std::cout << command << " returned with status " << ran << std::endl;
    }
  }
  return 0;
}
