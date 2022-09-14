import os
import torch
import numpy as np
import pandas as pd

from train.utils import *

def test(model, dataloader, csv_output:str):

    df = pd.DataFrame(columns=['image_path', 
                               'noise_type',
                               'noise_value',
                               'noise_score',
                               'predictions'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    loss_function = torch.nn.MSELoss()
    
    for idx, (x, img_path, ns_tp, ns_vl, y) in enumerate(dataloader):

        x = x.to(device)
        y = y.to(device)

        n = len(y)

        # predictions
        y_pred = model(x.float())
        y_pred = torch.reshape(y_pred, (n, 1, 1))
        loss = loss_function(y.float(), y_pred)

        # converting variables to list for DataFrame storing
        img_path = np.array(img_path).tolist()
        ns_tp = np.array(ns_tp).tolist()
        ns_vl = ns_vl.cpu().detach().numpy().tolist()
        ns_sc = y.cpu().detach().numpy().tolist()
        preds = y_pred.cpu().detach().numpy().tolist()

        d = {'image_path': img_path,
             'noise_type': ns_tp,
             'noise_value': ns_vl,
             'noise_score': ns_sc,
             'predictions': preds}

        sub_df = pd.DataFrame(data=d)

        df = pd.concat([df, sub_df], ignore_index=True)

        print(f"Testing iteration {idx+1} done \n")

        if idx % 5 == 0 : df.to_csv(csv_output)

    df.to_csv(csv_output)

        