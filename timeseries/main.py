from . import utils
from .loader import *
from .model import *

def main():
    classes = ['W','R','N1','N2','N3']
    npz_dir = utils.get_dir('data', 'npz')

    datapaths = split_data(dir=npz_dir, train_size=2, val_size=1, test_size=1)
    
    dataframes = create_dataframes(datapaths, exist=True)
    datasets = create_datasets(dataframes, seq_len=7680)

    train_dl, val_dl, test_dl = create_dataloaders(datasets, batch_size=8)

    model = Transformer(in_size=3,
                        out_size=len(classes),
                        nhead=1,
                        num_layers=1,
                        dim_feedforward=2048,
                        dropout=0)

if __name__ == "__main__":
    main()