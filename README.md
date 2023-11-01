# BASNet

## Environment setup
```ruby
cuda: 11.7  
python: 3.9  
pytorch: 1.13.1  
```
## Installation
```ruby
git clone https://github.com/JasonWZC/BASNet-main.git 
cd BASNet-main 
```
## Train
You can set the training model and parameters in `BASNet/train.py`.
```ruby
python train.py
```
## Prediction  
After training, you can put weight in `BASNet/predict/plot_demo.py`.  
Then use it to generate a prediction graph.
```ruby
python plot_demo.py
```
## Download the datesets
* LEVIR-CD:
  [LEVIR-CD](https://justchenhao.github.io/LEVIR/)

* CDD:
  [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)

* GZ-CD:
  [GZ-CD](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)

You can install these datasets to './datasets'. And we have provided a small dataset called TEST for your testing.

  
