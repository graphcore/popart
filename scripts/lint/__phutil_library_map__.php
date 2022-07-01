<?php
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

phutil_register_library_map(array(
  '__library_version__' => 2,
  'class' => array(
    'PreCommitLinter' => 'PreCommitLinter.php',
  ),
  'function' => array(),
  'xmap' => array(
    'PreCommitLinter' => 'ArcanistExternalLinter',
  ),
));
