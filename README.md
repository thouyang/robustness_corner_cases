# AI robustness analysis with consideration of corner cases

If you find this code helpful, please consider to cite our paper "AI robustness analysis with consideration of corner cases"

## Introduction
Corner cases are a set of high-risk data in deep learning (DL) systems, which could lead to incorrect and unexpected behaviors. 
To study corner cases' influence on DL models' robustness and stability, this research is implemented with corner case description and detection,
 as well as DL model's robustness analysis, as shown below. 

![DL robustness analysis](images/robustness_analysis_with_cc.png)

Firstly, a corner case descriptor based on surprise adequacy was introduced for corner case data detection, its high values are proved useful to reflect incorrect behaviors in classification. 
Then, based on the proposed corner case descriptor, training dataset was updated by removing data having high possibility to be corner case, and utilized for model retraining. A practical robustness analysis method was applied to measure the robustness radius of both the original and the retrained DL models. 

## Files and Directories

- `main.py` - Script for modeling training and robustness measurement.
- `cnnmodel.py` - CNN model.
- `mnistload.py` - MNIST data loading scripts.  
- `cc_remove.py` - Used for corner case detection and removing.
- `dif_dsa.py` -tools for DSA calculation.
- `deepfool.py` - tools for robustness measurement based on DeepFool

- `model` directory - Used for saving models and retrained models.
- `results` directory - Used for saving results, including robustness measurement, dsa calculation, corner case detection

### Command-line options of main.py

- `-batch_size` - Batch size. Default is 128.
- `-learning_rate` - Model learning rate. Default is 0.0002.
- `-n_classes` - The number of classes in dataset. Default is 10.
- `-num_epochs` - The number of epochs in training. Default is 20.
- `-is_retrain` - Selecting the process is model retraining process or not. Default is False.

## How to Use

This is a simple example of studying robustness measurement of DL based on MNIST dataset, as well as the robustness analysis with consideration of corner cases removing.

```bash
# train CNN model for mnist data and measure robustness
python main.py 

# detect corner case data
python cc_remove.py 

# model retraining with corner case removing and robustness measurement
python main.py -is_retrain True
```

### References
["Corner case data description and detection"](https://arxiv.org/abs/2101.02494) \
["Guiding Deep Learning System Testing using Surprise Adequacy"](https://arxiv.org/abs/1808.08444)\
["DeepFool: a simple and accurate method to fool deep neural networks"](https://arxiv.org/abs/1511.04599)
