import torch
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import librosa
import librosa.display

from .. import utils
from ..loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

def plot_signals(signals):
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 5 * len(signals)))
    
    for i, (X, X_dec) in enumerate(signals):
        X = X.cpu().numpy()
        X_dec = X_dec.cpu().numpy()
 
        X = X.reshape(-1)
        X_dec = X_dec.reshape(-1)

        axes[i].plot(X, label='Raw Signal', alpha=0.5)
        axes[i].plot(X_dec, label='Decoded Signal', alpha=0.5)
        axes[i].set_title(f'Signal Plot {i+1}')
        axes[i].set_xlabel('Samples')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()

    plt.tight_layout()

    path = utils.get_path('static', 'autoencoder', filename='signals_plot.png')
    plt.savefig(path)
    plt.close(fig)

def plot_concat_signals(signals):
    all_raw_signals = []
    all_decoded_signals = []

    for X, X_dec in signals:
        X = X.cpu().numpy().reshape(-1)
        X_dec = X_dec.cpu().numpy().reshape(-1)
        
        all_raw_signals.append(X)
        all_decoded_signals.append(X_dec)

    all_raw_signals = np.concatenate(all_raw_signals)
    all_decoded_signals = np.concatenate(all_decoded_signals)

    plt.figure(figsize=(100, 20))
    plt.plot(all_raw_signals, label='Raw Signal', alpha=0.5)
    plt.plot(all_decoded_signals, label='Decoded Signal', alpha=0.5)

    plt.title('Raw vs Decoded Signals')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()

    path = utils.get_path('static', 'autoencoder', filename='concat_signals_plot.png')
    plt.savefig(path)
    plt.close()

def test(data, criterion, model, fs, visualize=False):
    mfn = utils.get_path('models', filename='autoencoder.pth')

    model.load_state_dict(torch.load(mfn))
    model.to(device)
    model.eval()

    batches = len(data)
    total_test_loss = 0.0
    signals = []

    progress_bar = tqdm(enumerate(data), total=batches, desc=f'Evaluation', leave=True)

    with torch.no_grad():
        for _, (X, _) in progress_bar:
            X = X[:, :, :2].to(device)

            X_dec, latent = model(X)

            test_loss = criterion(X_dec, X)

            total_test_loss += test_loss.item()
            progress_bar.set_postfix(Loss=test_loss.item())

            signals.append((X, X_dec))

        if visualize:
            plot_signals(signals)
            plot_concat_signals(signals)

        avg_test_loss = total_test_loss / batches

    logger.info(f'\nTesting complete!\nTesting Loss: {avg_test_loss:.6f}\n')

def main():
    npz_dir = utils.get_dir('data', 'npz')
    
    samples, chunks = 7680, 32
    seq_len = samples // chunks
    
    datapaths = split_data(dir=npz_dir, train_size=1, val_size=1, test_size=1)
    
    _, _, test_df = get_dataframes(datapaths, rate=seq_len, exist=True)

    datasets = create_datasets(dataframes=(test_df,), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=512, drop_last=True)

    model = Autoencoder(seq_len=seq_len,
                        num_feats=2, 
                        latent_seq_len=1, 
                        latent_num_feats=8, 
                        hidden_size=8, 
                        num_layers=1,
                        dropout=0.5)
    
    test(data=dataloaders[0],
         criterion=utils.PowerLoss(p=2),
         model=model,
         fs=get_fs(path=datapaths),
         visualize=True)

if __name__ == '__main__':
    main()