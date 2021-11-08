# DGRL-Net
The code in this repository for paper "[Difference-Guided Representation Learning Network for Multivariate Time-Series Classification]" accepted by IEEE Transactions on Cybernetics.



## Dependencies

* tensorflow 1.14.0



## 18 MTS Datasets

The 18 MTS (Multivariate Time Series) benchmark data sets can download from [link](https://pan.baidu.com/s/1xxWMMqN5FrkbIWjsze_reg). They are collected from different repository, such as UCI, UCR and so on. These benchmark data sets come from various fields and have various input ranges and different numbers of classes, variables, and instances. 

And the folder `datasets` contain a demo dataset `ECG` .  The file `ECG.p`  is a list of three numpy arrays with `[samples, lables, original_lengths]`.  Meanwhile,

```
samples.shape = (number of instances, time length, number of variables)
labels.shape = (number of instances,)
original_lengths.shape = (number of instances,)
```



## Usage

You can run the command 
```
python AD.py
```
