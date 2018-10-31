CONFIGURE and BUILD
-------------------

This will be cleaned up.

On Ubunutu, I had success with
```
cmake -DONNX_DIR=../installonnx/share/cmake/ONNX 
      -DCMAKE_PREFIX_PATH=/home/jamesn/ws_willow/installpybind11/share/cmake/pybind11/ 
      -DPOPLAR_INSTALL_DIR=~/poplar_download/poplar-install/  
      ../willow/
```

where the above is run from directory buildwillow, 

jamesn@sw15-ublade00:~/ws_willow/buildwillow$ ls ..
buildonnx  buildpybind11  buildwillow  installonnx  installpybind11  onnx  pybind11  willow

install pytorch (use directions on website)

export LD_LIBRARY_PATH=~/ws_willow/willow/pywillow/:~/poplar_download/poplar-install/lib/

and you should be ready to use pywillow
