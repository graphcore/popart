// A tool for visualising the Ir (PopONNX's Intermediate representation).
// as a .dot file which can be compiled into a .pdf file.

#include <iostream>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/onnxutil.hpp>
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

int main(int argc, char **argv) {

  logging::ir::info("{} user command-line parameters received", argc - 1);

  int nArgsExpected = 4;
  if (argc - 1 != nArgsExpected) {
    std::stringstream ss;
    // The command-line parameters:
    ss << "\nExpected " << nArgsExpected << " command-line parameters:"
       << "\n\t (1) path of the ONNX model file"
       << "\n\t (2) path of the output directory"
       << "\n\t (3) compile dot->pdf(s) (0/1)"
       << "\n\t (4) maximum number of Ops to write to the .dot file"
       << "\n\t Example usage : ./dot_inference_graph /path/to/model.onnx . 1 "
          "121";
    // note for (3):
    // - unix only, uses std::system
    // - requires the dot program to be installed "./dot --help"
    //
    // note for (4):
    // - the first min(number of Ops in Ir, (4)) will be written
    std::cout << ss.str() << std::endl;
    return 0;
  }
  std::string modelPath = argv[1];
  logging::ir::info("modelPath set to {}", modelPath);
  std::string outputDir = argv[2];
  logging::ir::info("outputDir set to {}", modelPath);
  std::string argv3 = std::string(argv[3]);
  if (argv3 != "0" && argv3 != "1") {
    throw error("Expected arg 3 to be '0' or '1', not {}", argv3);
  }
  bool compileDotsToPDF = std::stoi(argv3);
  logging::ir::info("compileDotsToPDF set to {}", compileDotsToPDF);
  std::string argv4 = std::string(argv[4]);
  if (!is_number(argv4)) {
    throw error("Expected arg 4 to be a number, not {}", argv4);
  }
  SessionOptions opts;
  opts.finalDotOp = std::stoi(argv4);

  auto modelProto = onnxutil::getModelProto(modelPath);
  std::vector<std::string> dotStrings;

  // Only the FINAL .dot file. Append others here as required.
  opts.dotChecks.insert(DotCheck::FINAL);

  opts.logDir = outputDir;

  if (modelProto.graph().output_size() == 0) {
    throw error("Cannot generate dot for ONNX graph with no output");
  }

  auto out      = modelProto.graph().output(0).name();
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},      // in inference mode, so no losses,
              nullptr, // and no optimizer.
              opts,
              Patterns(PatternsLevel::NONE).enableInPlace(true)});

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
