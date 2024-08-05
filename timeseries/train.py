from . import utils
from . import loader

dfs_exist = False

def main():
    npz_dir = utils.get_dir('data', 'npz')

    data = loader.split_data(dir=npz_dir, train_size=57, val_size=1, test_size=1)
    
    train_df, val_df, test_df = loader.create_dataframes(*data, exist=False)

if __name__ == "__main__":
    main()