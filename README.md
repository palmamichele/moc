## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)

## Installation
To get started install the required dependencies.

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

Clone FMCA and set the DD branch (3487dad)
```bash
git clone https://github.com/muchip/fmca.git
cd fmca
git checkout DD 
cd ..
```

Compile FMCA following FMCA README.md  (later we use our cmake file)
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make    
cd .. 
```  


```bash
git clone https://github.com/palmamichele/moc
cd moc
mv ../fmca .    
```  



##Build in moc project folder
```bash
mkdir build
cd build
cmake ..
make     
```  

## Usage
for python see 
moc/moc_ECLIPSE.py file
