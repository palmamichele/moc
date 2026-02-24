## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)

## Installation
To get started, clone this repository and install the required dependencies.


```bash
git clone https://github.com/palmamichele/moc
cd moc         
```  

Install pybind11

```bash
git clone https://github.com/pybind/pybind11
```

Install Eigen (place in include the library: https://libeigen.gitlab.io)

```bash
mkdir include
cd include
```

Ensure OpenMP is available in your compiler (e.g. g++)


```bash
mkdir build
cd build
cmake .. \
  -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15
make     
```  

##Usage
run data.py file
