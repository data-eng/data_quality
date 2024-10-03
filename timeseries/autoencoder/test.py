import torch
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt

from .. import utils
from ..loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

def plot_signals(signals, outlier_threshold=1, dpi=1200):
    num_batches = len(signals)
    num_features = signals[0][0].shape[-1]
    
    for i, (X, X_dec) in enumerate(signals):
        fig, axes = plt.subplots(1, num_features, figsize=(10 * num_features, 5))

        for j in range(num_features):
            ax = axes[j] if num_features > 1 else axes

            X_feat = X[:, j].cpu().numpy().reshape(-1)
            X_dec_feat = X_dec[:, j].cpu().numpy().reshape(-1)

            outliers_X = np.where(np.abs(X_feat) > outlier_threshold)[0]
            outliers_X_dec = np.where(np.abs(X_dec_feat) > outlier_threshold)[0]

            X_feat[outliers_X] = 0
            X_dec_feat[outliers_X_dec] = 0

            ax.plot(X_feat, label='Raw Signal', color='C1', alpha=0.5, linewidth=0.2)
            ax.plot(X_dec_feat, label='Decoded Signal', color='C0', alpha=0.5, linewidth=0.2)

            ax.set_ylim(-outlier_threshold, outlier_threshold)
            ax.scatter(outliers_X, np.zeros_like(outliers_X), color='C1', label='Raw Outliers', s=30)
            ax.scatter(outliers_X_dec, np.zeros_like(outliers_X_dec), color='C0', label='Decoded Outliers', s=30)

            ax.set_title(f'Batch {i+1} - Feature {j+1}')
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')
            ax.legend()
        
        plt.tight_layout()

        path = utils.get_path('static', 'autoencoder', 'signals', filename=f'batch_{i+1}.png')
        plt.savefig(path, dpi=dpi)
        plt.close(fig)

def plot_concat_signals(signals, outlier_threshold=1):
    num_features = signals[0][0].shape[-1]

    all_raw_signals = [[] for _ in range(num_features)]
    all_decoded_signals = [[] for _ in range(num_features)]

    for X, X_dec in signals:
        for j in range(num_features):
            all_raw_signals[j].append(X[:, j].cpu().numpy().reshape(-1))
            all_decoded_signals[j].append(X_dec[:, j].cpu().numpy().reshape(-1))

    _, axes = plt.subplots(1, num_features, figsize=(10 * num_features, 5))
    
    for j in range(num_features):
        raw_signals_concat = np.concatenate(all_raw_signals[j])
        decoded_signals_concat = np.concatenate(all_decoded_signals[j])

        outliers_raw = np.where(np.abs(raw_signals_concat) > outlier_threshold)[0]
        outliers_decoded = np.where(np.abs(decoded_signals_concat) > outlier_threshold)[0]

        raw_signals_concat[outliers_raw] = 0
        decoded_signals_concat[outliers_decoded] = 0

        ax = axes[j]

        ax.plot(raw_signals_concat, label=f'Raw Signal', color='C1', alpha=0.5)
        ax.plot(decoded_signals_concat, label=f'Decoded Signal', color='C0', alpha=0.5)

        ax.set_ylim(-outlier_threshold, outlier_threshold)

        ax.scatter(outliers_raw, np.zeros_like(outliers_raw), color='C1', label='Raw Outliers', s=30)
        ax.scatter(outliers_decoded, np.zeros_like(outliers_decoded), color='C0', label='Decoded Outliers', s=30)

        ax.set_title(f'Signal - Feature {j+1}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        ax.legend()
    
    plt.tight_layout()
    
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
            plot_signals(signals, outlier_threshold=0.05)
            #plot_concat_signals(signals)

        avg_test_loss = total_test_loss / batches

    logger.info(f'\nTesting complete!\nTesting Loss: {avg_test_loss:.6f}\n')

def main():
    npz_dir = utils.get_dir('data', 'npz')
    
    samples, chunks = 7680, 32
    seq_len = samples // chunks
    
    datapaths = split_data(dir=npz_dir, train_size=46, val_size=3, test_size=10)
    
    _, _, test_df = get_dataframes(datapaths, rate=seq_len, exist=True)

    datasets = create_datasets(dataframes=(test_df,), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=512, drop_last=True)

    model = Autoencoder(seq_len=seq_len,
                        num_feats=2, 
                        latent_seq_len=1, 
                        latent_num_feats=8, 
                        hidden_size=64, 
                        num_layers=1,
                        dropout=0)
    
    test(data=dataloaders[0],
         criterion=utils.LogPowerLoss(p=1),
         model=model,
         fs=get_fs(path=datapaths),
         visualize=True)

if __name__ == '__main__':
    main()