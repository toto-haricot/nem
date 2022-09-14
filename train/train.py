import os
import time
import wandb
import torch
import numpy as np

from data import *
from utils import *


config = load_config("/workspace/nem/NEM/configs/configuration.yaml")

id_ = config['training']['training_id']
ckpt_path = config['training']['ckpt_path']

def train(model, path_train:list, path_val:list, n_epochs:int, learning_rate:float, checkpoints_path:str=ckpt_path):

    # datasets creation
    dataset_val = DatasetNoise(data=path_val, type_='train')
    dataset_train = DatasetNoise(data=path_train, type_='train')

    # dataloader creation
    dataloader_val = DataLoader(dataset=dataset_val, 
                        batch_size=config['training']['batch_size'],
                        shuffle=True)
    dataloader_train = DataLoader(dataset=dataset_train, 
                        batch_size=config['training']['batch_size'],
                        shuffle=True)

    # define on wich processor to train model
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device for training: {device}\n")

    # loss and optimizer for training
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    model.to(device)

    # avoiding overfitting X done directly with Adam optimizer
    # lr_scheduler = LRScheduler(optimizer=optimizer)


    # training loop
    for epoch in range(n_epochs):

        t1 = time.time()

        print(f"[TRAINING] Starting epoch {epoch+1}/{n_epochs}")
        print(f"[TRAINING] Device for training: {device}\n")

        epoch_train_loss = []
        epoch_train_accuracy = []

        for idx ,(x, _, y) in enumerate(dataloader_train):

            # print(f"\t\titeration {idx+1} epoch {epoch+1} time reading : {torch.sum(t1).item()} sec")
            # print(f"\t\titeration {idx+1} epoch {epoch+1} time croppping : {torch.sum(t2).item()} sec")
            # print(f"\t\titeration {idx+1} epoch {epoch+1} time noise creation : {torch.sum(t3).item()} sec")
            # print(f"\t\titeration {idx+1} epoch {epoch+1} time features computation : {torch.sum(t4).item()} sec")

            y = torch.reshape(y, (len(y), 1, 1))

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            t1_ = time.time()

            y_pred = model(x.float())
            y_pred = torch.reshape(y_pred, (len(y), 1, 1))

            loss = loss_function(y_pred, y.float())
            loss.backward()
            optimizer.step()

            # t2_ = time.time()
            # print(f"\t\titeration {idx+1} epoch {epoch+1} forward + backward time : {round(t2_-t1_,4)}")

            epoch_train_loss.append(loss.item())
            epoch_train_accuracy.append((loss.item())**.5)
            wandb.log({'epoch':epoch+1,
                       'iteration':idx+1,
                       'loss_train':loss.item(),
                       'mean_error_train':(loss.item())**.5})
        

            print(f"\t\titeration {idx+1} epoch {epoch+1} MSE : {loss.item():.4f}")
            print(f"\t\titeration {idx+1} epoch {epoch+1} accuracy : {(loss.item())**.5:.4f}\n")

        t2 = time.time()
        print(f"[TRAINING] Time for epoch {epoch+1} : {round(t2-t1, 4)} sec")

        # overall epoch loss on train set
        train_loss = mean(epoch_train_loss)
        train_accuracy = mean(epoch_train_accuracy)

        # compute validation loss
        epoch_val_loss = []
        epoch_val_accuracy = []

        print(f"[VALIDATION] Start validation epoch {epoch+1}/{n_epochs}")

        for idx ,(x, _, y) in enumerate(dataloader_val):

            y = torch.reshape(y, (len(y), 1, 1))

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x.float())
            y_pred = torch.reshape(y_pred, (len(y), 1, 1))

            loss_val = loss_function(y_pred, y.float()).item()

            epoch_val_loss.append(loss_val)
            epoch_val_accuracy.append(loss_val**.5)

        val_loss = mean(epoch_val_loss)
        val_accuracy = mean(epoch_val_accuracy)

        wandb.log({'epoch':epoch+1, 
                   'loss_val':val_loss, 
                   'accuracy_val':val_accuracy})

        print(f"\t\tepoch {epoch+1} validation loss : {val_loss:.4f}")
        print(f"\t\tepoch {epoch+1} validation  accuracy : {val_accuracy:.4f}\n")
            
        # we use adam optimizer so we should not update again the lr
        # lr_scheduler(val_loss)

        # saving model weights
        ckpt_name = os.path.join(ckpt_path, "nem_training_"+id_+f"_epoch_{epoch}.ckpt")
        
        if epoch >= 10: 
            ckpt_old = os.path.join(ckpt_path, "nem_training_"+id_+f"_epoch_{epoch-10}.ckpt")
            os.remove(ckpt_old)

        torch.save(model.state_dict(), ckpt_name)

    ckpt_name = os.path.join(ckpt_path, f"nem_training_"+id_+".ckpt")
    torch.save(model.state_dict(), ckpt_name)


    
