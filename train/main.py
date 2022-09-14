import wandb
import numpy as np
import pandas as pd

from data import *
from model import *
from train import *
from utils import *

def main():

    # getting configuration parameters 
    config = load_config("/workspace/nem/NEM/configs/configuration.yaml")

    # keeping track of model performances and parameters
    wandb.init(project="NEM", 
               entity="toto-haricot", 
               name="training_"+config['training']['training_id'],
               notes="rm brisque ftrs // nrmlzt on 100 of kurt ftrs // ")
    
    wandb.config = {
        'learning_rate':config['training']['learning_rate'],
        'epochs':config['training']['n_epochs'],
        'batch_size':config['training']['batch_size'],
        'noise_creation': config['noise_creation'],
        'mlp_architecture': config['model']['layers']}

    # spliting paths between train, test and val set with same shooting-id
    paths_train, paths_test, paths_val = train_test_split(config['dataset']['csv_path'])

    # paths_train = pd.read_csv("/workspace/nem/NEM/configs/paths_train.csv", index_col=0)
    # paths_test = pd.read_csv("/workspace/nem/NEM/configs/paths_test.csv", index_col=0)
    # paths_val = pd.read_csv("/workspace/nem/NEM/configs/paths_val.csv", index_col=0)

    # paths_train = np.array(paths_train).tolist()
    # paths_test = np.array(paths_test).tolist()
    # paths_val = np.array(paths_val).tolist()

    # saving paths used for training, testing and validation
    pd.DataFrame(paths_val).to_csv("/workspace/nem/NEM/configs/paths_val.csv")
    pd.DataFrame(paths_test).to_csv("/workspace/nem/NEM/configs/paths_test.csv")
    pd.DataFrame(paths_train).to_csv("/workspace/nem/NEM/configs/paths_train.csv")
    
    # model instance creation
    model = NeuralNetwork(layers_structure=config['model']['layers'])

    train(model=model,
        path_train=paths_train,
        path_val=paths_val,
        n_epochs=config['training']['n_epochs'],
        learning_rate=config['training']['learning_rate'])

if __name__=='__main__':
    main()
