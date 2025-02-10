# CudaSHA256
Simple tool to calculate sha256 on GPU using Cuda

# Built
```
nvcc main.cu
```

# Run
```
./a.out <some test file> <another test file> ...
or
nvprof ./a.out <some test file> <another test file> ...
```


# Notes
The hash function is correct. Rest of code can be kept for testing, but otherwise needs to be changed
Is this parallelized or for computation of a single hash serially?