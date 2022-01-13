<?php
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

final class PopartTestLinter extends ArcanistExternalLinter {

  public function getInfoName() {
    return 'Popart test linter';
  }

  public function getLinterName() {
    return 'popart-test-linter';
  }

  public function getLinterConfigurationName() {
    return 'popart-test-linter';
  }

  public function getDefaultBinary() {
    return dirname(__FILE__) . '/../../check_test_has_cmake_entry.py';
  }

  // An error is thrown if this method is not present. 
  public function getInstallInstructions() {
    return pht('');
  }

  // This gets the linter to run `check_test_has_cmake_entry.py` with python.
  public function shouldUseInterpreter() {
    return true;
  }

  // This gets the linter to run `check_test_has_cmake_entry.py` with python.
  public function getDefaultInterpreter() {
    return 'python3';
  }

  // `check_test_has_cmake_entry.py` uses return code 1 to indicate that the lint check failed.
  public function shouldExpectCommandErrors() {
    return true;
  }

  protected function parseLinterOutput($path, $err, $stdout, $stderr) {
    if ($err == 0) {
      return array();
    }

    $message = id(new ArcanistLintMessage())
      ->setPath($path)
      ->setName($this->getLinterName())
      ->setSeverity(ArcanistLintSeverity::SEVERITY_ERROR)
      ->setDescription($stderr);
    return array($message);
  }
}
