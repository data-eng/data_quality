from . import utils
from . import loader

def main():
    npz_dir = utils.get_dir('data', 'npz')

    datapaths = loader.split_data(dir=npz_dir, train_size=2, val_size=1, test_size=1)
    
    dataframes = loader.create_dataframes(*datapaths, exist=True)
    datasets = loader.create_datasets(*dataframes, seq_len=7680)

if __name__ == "__main__":
    main()