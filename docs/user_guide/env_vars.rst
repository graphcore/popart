Environment variables
---------------------

POPART_LOG_LEVEL
~~~~~~~~~~~~~~~~~

TODO

POPART_LOG_DEST
~~~~~~~~~~~~~~~~

TODO

POPART_LOG_CONFIG
~~~~~~~~~~~~~~~~~~

TODO

POPART_DOT_CHECKS
~~~~~~~~~~~~~~~~~~

Supported values:

- FWD0
- FWD1
- BWD0
- PREALIAS
- FINAL

These values may be combined using `:` as a separator.
The below shows how to set `POPART_DOT_CHECKS` to export
dot graphs `FWD0` and `FINAL`.

::

  export POPART_DOT_CHECKS=FWD0:FINAL

The values of `POPART_DOT_CHECKS` will be combined with any values
that are found in the session options.

POPART_TENSOR_TILE_MAP
~~~~~~~~~~~~~~~~~~~~~~~

The tensor tile map may be saved to a file by setting this.
The value this is set to will be used for the file name of the tensor tile map.
The format of the tensor tile map is json.
The tensor tile map will be saved in the call to `Session::prepareDevice`.
The below show how to set the variable to save the tensor tile map to `ttm.js`.

::

  export POPART_TENSOR_TILE_MAP=ttm.js
