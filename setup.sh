#!/bin/bash
# Setup environment for OpenFHE Iris Project

sudo apt update
sudo apt install -y cmake g++ make python3-venv python3-pip

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
