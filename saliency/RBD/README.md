# Saliency Detection
Code migrating from <https://github.com/MCG-NKU/SalBenchmark/tree/master/Code/matlab/RBD> with minor fixes.

This code is tested on Ubuntu with `gcc (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609` and `Matlab R2014b (8.4.0.150421) 64-bit (glnxa64)`.

## Mex compilation:
```
cd Funcs/SLIC
mex -v CXXFLAGS="\$CXXFLAGS -std=c++11"  SLIC_mex.cpp SLIC.cpp
```

