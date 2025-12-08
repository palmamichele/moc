## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)


## Installation
To get started, clone this repository and ensure eigen library is available.
Ensure clang with openMP support is enabled (e.g. from homebrew)

```bash
cd moc
clang++ -O3 -std=c++17 -Xpreprocessor -fopenmp -fPIC \
  -I include/eigen \
  -c bprefix_lib.cpp -o bprefix_lib.o

clang++ -O3 -std=c++17 -Xpreprocessor -fopenmp -fPIC \
  -I include/eigen \
  -c lsh_moc.cpp -o lsh_moc.o


clang++ -dynamiclib bprefix_lib.o lsh_moc.o \
  -L/opt/homebrew/opt/libomp/lib -lomp \
  -Wl,-rpath,/opt/homebrew/opt/libomp/lib \
  -o libcombined.dylib
```

## Usage
```bash
python main_c.py
```







