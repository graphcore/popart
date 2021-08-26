<?php
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

final class AutoPatchLinter extends ArcanistExternalLinter
{
  private $script_path;

  public function getInfoName()
  {
    return 'PopART Linter Orchestrator';
  }

  public function getInfoURI()
  {
    return '';
  }

  public function getInfoDescription()
  {
    return 'Custom Python linter which handles the application of patches produced by other linters';
  }

  public function getLinterName()
  {
    return 'autopatch';
  }

  public function getLinterConfigurationName()
  {
    return 'autopatch';
  }

  public function getLinterConfigurationOptions()
  {
    return parent::getLinterConfigurationOptions();
  }

  public function getDefaultBinary()
  {
    return  dirname(__FILE__) . '/../run_lint.py';
  }

  public function getInstallInstructions() {
    return pht('This linter is included with PopART under `scripts/`.');
  }

  public function shouldUseInterpreter()
  {
    return true;
  }

  public function getDefaultInterpreter()
  {
    return 'python3';
  }

  public function shouldExpectCommandErrors()
  {
    return false;
  }

  protected function getMandatoryFlags()
  {
    return array();
  }

  protected function parseLinterOutput($path, $err, $stdout, $stderr)
  {
    $linter_output = json_decode($stdout, true);
    $ok = $linter_output['status_ok'];

    $patch_msg = '';
    foreach ($linter_output['messages'] as $linter_msg) {
      $patch_msg .= sprintf("[%s]: %s\n", $linter_msg['linter'], $linter_msg['message']);
    }

    $message = id(new ArcanistLintMessage())
      ->setPath($path)
      ->setGranularity(ArcanistLinter::GRANULARITY_FILE)
      ->setName($this->getLinterName())
      ->setDescription($patch_msg);
    if ($ok) {
      $orig = $linter_output['original'];
      $new = $linter_output['replacement'];
      if ($orig == $new) {
        return array();
      }
      $message->setSeverity(ArcanistLintSeverity::SEVERITY_AUTOFIX)
        ->setLine(1)
        ->setChar(1)
        ->setOriginalText($orig)
        ->setReplacementText($new);
    } else {
      $message->setSeverity(ArcanistLintSeverity::SEVERITY_ERROR);
    }
    return array($message);
  }
}
