import os
import torch
import argparse

from train.data import *
from train.utils import *
from train.model import *

from test import *

def main():

    # ----------------------------------------------------------------------------------------------------
    # 1. ARGUMENT PARSING --------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, help="path to chekpoints")
    parser.add_argument('--folder_path', type=str, help="path to images folder")

    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    folder_path = args.folder_path

    # ----------------------------------------------------------------------------------------------------
    # 2. LOADING MODEL -----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------

    model = NeuralNetwork(input_size=65)

    config = load_config("/workspace/nem/NEM/configs/configuration.yaml")

    if ckpt_path is None : ckpt_path = config['testing']['ckpt_path']

    print(f"testing with checkpoints from {ckpt_path}")

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt, strict=True)

    # ----------------------------------------------------------------------------------------------------
    # 3. RUNNING PREDICTIONS -----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------

    paths_test = config['testing']['paths_test']

    paths_test = pd.read_csv(paths_test, index_col=0).values.tolist()

    dataset = DatasetNoise(data=paths_test, type_='test')
    dataloader = DataLoader(dataset=dataset, batch_size=config['training']['batch_size'], shuffle=True)

    test(model=model, dataloader=dataloader, csv_output="/workspace/nem/NEM/test/testing_results.csv")


# ----------------------------------------------------------------------------------------------------
# X. RUNNING MAIN ------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
