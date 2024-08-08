import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from .. import utils
from .loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

def test(data, criterion, model):
    mfn = utils.get_path('models', filename='autoencoder.pth')

    model.load_state_dict(torch.load(mfn))
    model.to(device)
    model.eval()

    batches = len(data)
    total_test_loss = 0.0

    progress_bar = tqdm(enumerate(data), total=batches, desc=f'Evaluation', leave=True)

    with torch.no_grad():
        for _, (X, _) in progress_bar:
            X = X[:, :, :2].to(device)

            X_dec, _ = model(X)

            test_loss = criterion(X_dec, X)

            total_test_loss += test_loss.item()
            progress_bar.set_postfix(Loss=test_loss.item())

        avg_test_loss = total_test_loss / batches

    logger.info(f'\nTesting complete!\nTesting Loss: {avg_test_loss:.6f}\n')

def main():
    npz_dir = utils.get_dir('data', 'npz')
    
    samples, chunks = 7680, 32
    seq_len = samples // chunks
    
    datapaths = split_data(dir=npz_dir, train_size=1, val_size=1, test_size=1)
    
    _, _, test_df = get_dataframes(datapaths, rate=seq_len, exist=False)

    datasets = create_datasets(dataframes=(test_df,), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=512)

    model = Autoencoder(seq_len=240,
                        num_feats=2, 
                        latent_seq_len=1, 
                        latent_num_feats=8, 
                        hidden_size=128, 
                        num_layers=2)
    
    test(data=dataloaders,
         criterion=nn.MSELoss(),
         model=model)

if __name__ == '__main__':
    main()