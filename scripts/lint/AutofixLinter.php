<?php
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

final class AutofixLinter extends ArcanistExternalLinter
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
    return 'autofix';
  }

  public function getLinterConfigurationName()
  {
    return 'autofix';
  }

  public function getLinterConfigurationOptions()
  {
    $options = array(
      'autofix.script_path' => array(
        'type' => 'optional string',
        'help' => pht('Pass in the script_path name.'),
      ),
    );

    return $options + parent::getLinterConfigurationOptions();
  }

  public function getDefaultBinary()
  {
    return  dirname(__FILE__) . '/../run_lint.py';
  }

  public function getInstallInstructions()
  {
    if ($this->script_path !== null) {
      $s = "Could not locate the linter script " . dirname(__FILE__)  . '/' . $this->script_path;
      return pht($s);
    }
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

  public function setLinterConfigurationValue($key, $value)
  {
    switch ($key) {
      case 'gc-copyright.script_path':
        $this->script_path = $value;
        return;
      default:
        return parent::setLinterConfigurationValue($key, $value);
    }
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
