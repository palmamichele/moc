#!/bin/bash

rm -rf experiments
rm -rf data

source .venv/bin/activate
python3 linear-california.py
python3 california.py
python3 iris.py
python3 MNIST.py
sleep 10
python3 mocs.py
python3 mini-batch-test.py
python3 alexnet.py
sleep 10
python3 mini-batch-alexnet.py

