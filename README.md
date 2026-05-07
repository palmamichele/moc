## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)

## Installation
It is recommended to create a virtual environment before installing dependencies.

## Create a virtual environment
```bash
python -m venv venv
pip install -r requirements.txt
```

## Ensure OpenMP is available in your compiler configured in CMAKE (e.g. g++)
```bash
export CC=/opt/homebrew/bin/gcc-15  
export CXX=/opt/homebrew/bin/g++-15
```


## Install pybind11 (sugg. via homebrew )

```bash
brew install pybind11
```

## Install Eigen (sugg. via homebrew)

```bash
brew install eigen
```


## Install ECLipsE 
```bash
git clone https://github.com/YuezhuXu/ECLipsE.git
cd ECLipsE
pip install -e .
cd ..
```

## Install FMCA and set the DD branch
```bash
git clone https://github.com/muchip/fmca.git
cd fmca
git checkout DD 
```


## Compile FMCA (following FMCA README.md)  (later we might use our cmake file)
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make    
cd .. 
```  


## Download your ImageNet dataset 
Refer to https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data (suggested kagglehub)

Convert the val folder to the same format as the train folder (i.e., move the images to their respective folders). (credits: https://github.com/fh295/semanticCNN, https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Usage
Grant permission and run the .sh script for reproducing paper results
```bash
chmod +x run.sh
./run.sh
```  
