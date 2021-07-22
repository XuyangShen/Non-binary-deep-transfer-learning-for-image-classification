# Non-binary deep transfer learning for image classification

This repository is the official implementation of [Non-binary deep transfer learning for image classification](https://arxiv.org/pdf/2107.08585.pdf). 

```python
__version__ = '0.1.0'
```



One of main findings: Optimal settings versus best default settings. 

![final_reult](./docs/final_results.png)

Left to right: Caltech, Cars, Aircraft, DTD.


#### Citing
```citing
```



## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

* The experiments were only verified in `torch 1.7.0` and later version. 

* Each individual experiment ran on one `Nvidia V100` GPU and 12 cores from one `24-core Intel Xeon` processor.

* Each final experiment ran on four `Nvidia V100` GPU and two `24-core Intel Xeon` processor.



## Datasets

Download the datasets and put in the folder [data](./data).

```datasets
- Imagenet 1K
- Caltech 256-30
- Stanford Cars
- FGVC Aircraft
- Describable Textures
```



## Training and Evaluation

To get the results in the paper,  adjust the following paramaters and fine-tune the [Inception v4 model](./timm/models/inception_v4.py). 


```python
python train.py \
    # dataset
--dataset \ # selec one of [imagenet, aircraft, dtd, caltech, car]
--data_dir "./data/" \ # path where the data stores
--num-classes \ # number of label classes
    # model
--model "inception_v4" \ # select the model Inception v4
--pretrained \ # load pre-trained weights
--new-layers=1 \ # number of layers to reinitialize when fine-tuning
    # hyper-para
-b 32 \ # batch size
--epochs 800 \ # epochs
--lr 0.03 \ # can be a single lr for all layers, or 3 lrs for different parts of layers --lr 0.03 0.03 0.03
--num-lowlrs=8 \ # must indicate if multiple learing rates is enabled, number of layers to train at a higher learning rate
--weight-decay 0.0005 \ # weight decay
--decay-epochs 10 \ # epoch interval to decay lr, default scheduler is step
--decay-rate 0.9 \ # lr decay rate
--mixup 0.2 \ # mixup alpha
--L2SP \ # Use L2SP weight decay
--L2SP_rates 0.1 \ # can be a L2SP rates for all layers, or multiple rates for different parts of layers --L2SP_rates 0.1 0.001 0.0
--aa rand-m9-mstd0.5-inc1 \ # AutoAugment policy
--reprob 0.5 \ # random erase prob
--remode pixel \ # random erase mode
    # output
--output "./experiments/..." \ # path to save checkpoints and results 
    # others
-j 16 \ # how many training processes to use 
--amp \ # enable mixed precision training
--native-amp \ # Use Native Torch AMP mixed precision
```




## Results

All results and evluation can be found in `Section 4 Results` in paper.



## Licenses

### Code
Our experiments were built based on [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models).
