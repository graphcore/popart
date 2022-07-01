<?php
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

final class PreCommitLinter extends ArcanistExternalLinter {

  public function getInfoName()
  {
    return 'Linter which calls pre-commit';
  }

  public function getLinterName() {
    return 'pre-commit';
  }

  public function getLinterConfigurationName() {
    return 'pre-commit';
  }

  public function getInfoDescription()
  {
    return 'Custom linter which calls pre-commit';
  }

  public function getDefaultBinary() {
    return 'pre-commit';
  }

  protected function getPathArgumentForLinterFuture($path) {
    return 'run --file ' . $path;
  }

  public function getInstallInstructions() {
    return pht('pip3 install pre-commit && pre-commit install && pre-commit run all');
  }

  public function shouldExpectCommandErrors()
  {
    return false;
  }

  protected function parseLinterOutput($path, $err, $stdout, $stderr) {
    if(! $err) return [];
    return false;
  }
}
