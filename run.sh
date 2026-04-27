#!/bin/bash

rm -rf experiments
rm -rf data

source .venv/bin/activate
python3 linear-california.py
sleep 10
python3 california.py
sleep 10
python3 iris.py
sleep 10
python3 MNIST.py
sleep 10
python3 mocs.py
sleep 10
python3 mini-batch-test.py

