# Noisesniffer
Source code of the article ''Noisesniffer: a Fully Automatic Image Forgery Detector Based on Noise Analysis" published in IWBF 2021.

# About the forgery-detection method
Images undergo a complex processing chain from the moment light reaches the camera’s sensor until the final digital image is delivered. Each of these operations leave traces on the noise model which enable forgery detection through noise analysis. In this article we define a background stochastic model which makes it possible to detect local noise anomalies characterized by their number of false alarms. The proposed method is both automatic and blind, allowing quantitative and subjectivity-free detections. Results show that the proposed method outperforms the state of the art.

<p align="center"> <img src="https://user-images.githubusercontent.com/47035045/116679770-39d9aa00-a9ab-11eb-826b-bf73690f08c8.png" width="50%"> </p>

# How to run the code

## Install the requirements
The libraries needed to run the program are listed in the file requirements.txt.
You can do the following to install them in a virtual environment (venv):

Install Python 3 and upgrade pip:
`sudo apt-get update`
`sudo apt-get install -y python3 python3-dev python3-pip python3-venv`
`pip install --upgrade pip`

Create the venv, activate it, and install the requirements:
`python3 -m venv ./venv`
`source ./venv/bin/activate`
`pip3 install -r requirements.txt`

## Run the code
Activate the venv:
`source ./venv/bin/activate`

Run the code with the image to analyze as argument:
`./Noisesniffer.py <input image>`

NoiseSniffer will create a folder (“results/”) which contains the results.

Example:

`source ./venv/bin/activate`
`./Noisesniffer.py images/img00.jpg`
After the execution the “results/” folder will contain a subdirectory 'img00/' containing the mask (mask_thresh1.png) and the NFA values (NFA_w5_W256_n0.05_m0.3_b20000.txt).

