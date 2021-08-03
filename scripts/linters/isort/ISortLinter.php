<?php

/**
 * Uses isort to sort Python imports.
 */
final class ISortLinter extends ArcanistExternalLinter {

  public function getInfoURI() {
    return 'https://pycqa.github.io/isort/';
  }

  public function getInfoDescription() {
    return pht('Uses isort to sort Python imports.');
  }

  public function getInfoName() {
    return 'isort';
  }

  public function getLinterName() {
    return 'isort';
  }

  public function getVersion() {
    list($stdout) = execx('python -c "import isort; print(isort.__version__)"');
    $matches = array();
    if (preg_match('/(?P<version>\d+\.\d+\.\d+)$/', $stdout, $matches)) {
      $version = $matches['version'];
      return $version;
    } else {
      return false;
    }
  }

  public function getLinterConfigurationName() {
    return 'isort';
  }

  public function getLinterConfigurationOptions() {
    $options = array(
    );

    return $options + parent::getLinterConfigurationOptions();
  }

  public function getDefaultBinary() {
    return 'isort';
  }

  public function getInstallInstructions() {
    return pht('Run "pip install isort" in your Python environment.');
  }

  public function shouldExpectCommandErrors() {
    return false;
  }

  protected function getMandatoryFlags() {
    // Force the formatted file to stdout.
    return array("--stdout");
  }

  protected function parseLinterOutput($path, $err, $stdout, $stderr) {
    $ok = ($err == 0);

    if (!$ok) {
      return false;
    }

    $root = $this->getProjectRoot();
    $path = Filesystem::resolvePath($path, $root);
    $orig = file_get_contents($path);
    if ($orig == $stdout) {
      return array();
    }

    $message = id(new ArcanistLintMessage())
      ->setPath($path)
      ->setLine(1)
      ->setChar(1)
      ->setGranularity(ArcanistLinter::GRANULARITY_FILE)
      ->setCode($this->getLinterName())
      ->setSeverity(ArcanistLintSeverity::SEVERITY_AUTOFIX)
      ->setName('Wrong import order.')
      ->setDescription("'$path' has code style errors.")
      ->setOriginalText($orig)
      ->setReplacementText($stdout);
    return array($message);
  }

}
