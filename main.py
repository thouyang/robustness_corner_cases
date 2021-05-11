import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import pandas as pd
import os
from dif_dsa import *
from mnistload import *
from cnnmodel import ConvNet
from rob_measure import rob_measure
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--learning_rate", "-learning_rate", help="Learning rate", type=float, default=0.0002
    )
    parser.add_argument(
        "--num_epochs", "-num_epochs", help="Number of epochs in training", type=int, default=20
    )
    parser.add_argument(
        "--n_class", "-n_class", help="Number of classes", type=int, default=10
    )
    parser.add_argument(
        "--n_times", "-n_times", help="Number of times for model training", type=int, default=50
    )
    parser.add_argument(
        "--is_retrain", "-is_retrain", help="Selecting if retraining the model", type=bool, default=False
    )

    args = parser.parse_args()
    TrainD, TestD,D0= data_pre(batch_size=args.batch_size,retrain=args.is_retrain)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for num_m in range(args.n_times):
        model = ConvNet(n_class=args.n_class).cuda()
        model.train(TrainD,num_epochs=args.num_epochs,learning_rate=args.learning_rate)
        if args.retrain:
            filename='re_'+str(num_m)
        else:
            filename = str(num_m)
    ###save model
        path='model/model_'+filename+'.h5'
        torch.save(model, path)
    ###robustness measurement
        path="results/rob_"+filename+'.csv'
        mean_radius=rob_measure(model,TestD,path)

    #### dsa calculation
        if not args.is_retrain:
            train_ats = torch.tensor([])
            train_pred = []
            y_train = []
            for images, labels in D0:
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data.cpu(), 1)
                train_ats = torch.cat((train_ats, outputs.data.cpu()), dim=0)
                train_pred += predicted.tolist()
                y_train += labels.tolist()
            #
            class_matrix, all_idx = cal_cla_matrix(y_train)
            train_ats = np.array(train_ats.data.cpu())
            test_dsa = cal_dsa3(train_ats, y_train, train_ats, train_pred, class_matrix)

            ### save dsa data
            mdic = {'ats': train_ats, 'dsa': np.array(test_dsa)}
            scio.savemat("results/dsa_" + filename + '.mat', mdic)
