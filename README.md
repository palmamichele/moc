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

Install pybind11 (sugg. via homebrew )

```bash
brew install pybind11
```

Install Eigen (sugg. via homebrew)

```bash
brew install eigen
```

Ensure OpenMP is available in your compiler configured in CMAKE (e.g. g++)
```bash
export CC=/opt/homebrew/bin/gcc-15  
export CXX=/opt/homebrew/bin/g++-15
```


Clone FMCA in the project folder (https://github.com/muchip/fmca.git) and set the branch
```bash
git clone https://github.com/muchip/fmca.git
cd fmca
git checkout DD 
```


Compile FMCA following FMCA README.md 
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make     
```  


##Usage
run moc/moc_ECLIPSE.py file
