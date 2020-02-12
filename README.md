# face_alignment_cpp

2D and 3D Face alignment library build using pytorch https://www.adrianbulat.com c++ implementation

## Contents

1. [Requirements](#requirements)
2. [Build](#build)
3. [Usage](#usage)


## Requirements

- Pytorch (tag: pytorch v1.4)
- Libtorch
- OpenCV

## Build

### Step 1

Export your pytorch model to torch script file, We will simply use resnet50 in this demo

### Step 2

Write your C++ program, check the file ``prediction.cpp`` for more detial.  

PS: ``module->to(at::kCUDA)`` and ``input_tensor.to(at::kCUDA)`` will switch your model & tensor to GPU mode,  
comment out them if you just want to use CPU mode. 


### Step 3

Write a ``CMakeLists.txt``, the version of OpenCV must the same as your libtorch.
Otherwise, you may get the compile error:

```
error: undefined reference to `cv::imread(std::string const&, int)'
```

check [issues 14684](https://github.com/pytorch/pytorch/issues/14684) and [issues 14620](https://github.com/pytorch/pytorch/issues/14620) for more details.

## Usage

- run ``model_trace.py``,   then you will get a file ``resnet50.pt``
- compile your cpp program, you need to use ``-DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch``, for example:

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/cgilab/pytorch/torch/lib/tmp_install ..
make
```

- test your program

``classifier <path-to-exported-script-module> <path-to-lable-file>``

```

