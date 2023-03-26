# Energy Disaggregation. A Case Study of Large Number of Components

This repository contains the source code to reproduce experiments 


![Alt text](images/architecture.jpg?raw=true "Title")


## Presentation and report

The presentation slides and reports are 


## Environments 

The implementation is GPU-based. Double GPU (GTX RTX 2080 Ti) is enough to train and test the models. All the calculations take approximately 40 minutes.

The list of the packages on which the mode tested is `requirements.txt` file. The most important one are

`torch==1.13.1 torchvision==0.4.1 numpy==1.21.6 timm==0.6.12 scikit-learn=1.1.2`


## Signal separation 

If you want to prepare Plug Load Appliance Identification Dataset (PLAID) dataset data from a scratch, you need to download it from the source [PLAD data](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619). Aggregate and sub-metered signals should be saved into `data/aggregated` and `data/submetered`, respectevily. <br />

Another option is to use already prepared data and run the whole jupyter notebook.


## Training and Testing models 

All the training and testing of ICAResNetFFN model and baseline Neural Network models (Temporal Pooling NILM and FIT-PS LSTM), you need to run jupyter notebook `jupyter notebook`. 


## Baseline classical ML models

To get the scores of baseline classical models for comparison, run the following command 
```
python baseline_classical_ml.py
```


## Results 






