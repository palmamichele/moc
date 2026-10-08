## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)

## Installation
It is recommended to create a virtual environment before installing dependencies.
suggested python >=3.14

## Create a virtual environment
```bash
python -m venv venv
pip install -r requirements.txt
```

## Install PGD-MOC
```bash
git clone https://github.com/palmamichele/pgdmoc.git
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
