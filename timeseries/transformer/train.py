import time
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

from .. import utils
from ..loader import *
from .model import *
from ..autoencoder import autoencoder as Autoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

def autoencode(X, y):
    """
    Applies autoencoder to the input and target tensors.

    :param X: tensor (batch_size, seq_len, num_feats)
    :param y: tensor (batch_size, seq_len, 1)
    :return: tuple of (X, y) tensors
    """
    channels, time = separate(src=X, c=[0,1], t=[3])

    _, channels = Autoencoder(channels)
    time = aggregate_seqs(data=time)

    X = merge(channels, time)
    y = aggregate_seqs(data=y)

    return X, y
    
def train(data, classes, chunks, epochs, patience, batch_size, lr, criterion, model, optimizer, scheduler, visualize=False):
    model.to(device)

    train_data, val_data = data

    batches = len(train_data)
    num_classes = len(classes)

    logger.info(f"Number of training iterations per epoch: {batches // (chunks * batch_size)}.")

    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(*scheduler, optimizer)

    best_val_loss = float('inf')
    stationary = 0
    train_losses, val_losses = [], []

    checkpoints = {'epochs': 0, 
                   'best_epoch': 0, 
                   'best_train_loss': float('inf'), 
                   'best_val_loss': float('inf'), 
                   'precision_micro': 0, 
                   'precision_macro': 0, 
                   'precision_weighted': 0,
                   'recall_micro': 0, 
                   'recall_macro': 0, 
                   'recall_weighted': 0, 
                   'fscore_micro': 0,
                   'fscore_macro': 0, 
                   'fscore_weighted': 0}

    for epoch in range(epochs):
        start = time.time()

        model.train()

        c, b = 1, 1
        total_train_loss = 0.0
        true_values, pred_values = [], []
        X_c, y_c, X_b, y_b = [], [], [], []

        progress_bar = tqdm(enumerate(train_data), total=batches, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        #for i, (X, y) in enumerate(train_data):
        for i, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)

            X, y = autoencode(X, y)

            X_c.append(X)
            y_c.append(y)

            if c == chunks or i == len(train_data) - 1:
                X = torch.cat(X_c, dim=1)
                y = torch.cat(y_c, dim=1)

                X_b.append(X)
                y_b.append(y)

                if b == batch_size:
                    X = torch.cat(X_b, dim=0)
                    y = torch.cat(y_b, dim=0)

                    perm = torch.randperm(X.size(0))
                    X, y = X[perm], y[perm]
   
                    y_pred = model(X)

                    batch_size, seq_len, _ = y_pred.size()
                    assert b == batch_size, f"Batch size mismatch: expected {b}, got {batch_size}"

                    y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
                    y = y.reshape(batch_size * seq_len)

                    train_loss = criterion(y_pred, y)

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    total_train_loss += train_loss.item()
                    progress_bar.set_postfix(Loss=train_loss.item())

                    true_values.append(y.cpu().numpy())
                    pred_values.append(y_pred.detach().cpu().numpy())

                    b, X_b, y_b = 1, [], []

                else:
                    b += 1

                c, X_c, y_c = 1, [], []

            else:
                c += 1

        avg_train_loss = total_train_loss / batches
        train_losses.append(avg_train_loss)

        model.eval()

        c, b = 1, 1
        total_val_loss = 0.0
        X_c, y_c, X_b, y_b = [], [], [], []

        with torch.no_grad():
            for j, (X, y) in enumerate(val_data):
                X, y = X.to(device), y.to(device)

                X, y = autoencode(X, y)

                X_c.append(X)
                y_c.append(y)

                if c == chunks or j == len(val_data) - 1:
                    X = torch.cat(X_c, dim=1)
                    y = torch.cat(y_c, dim=1)

                    X_b.append(X)
                    y_b.append(y)

                    if b == batch_size:
                        X = torch.cat(X_b, dim=0)
                        y = torch.cat(y_b, dim=0)

                        perm = torch.randperm(X.size(0))
                        X, y = X[perm], y[perm]

                        y_pred = model(X)

                        batch_size, seq_len, _ = y_pred.size()
                        assert b == batch_size, f"Batch size mismatch: expected {b}, got {batch_size}"

                        y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
                        y = y.reshape(batch_size * seq_len)

                        val_loss = criterion(pred=y_pred, true=y)
                        total_val_loss += val_loss.item()

                        b, X_b, y_b = 1, [], []

                    else:
                        b += 1

                    c, X_c, y_c = 1, [], []

                else:
                    c += 1
        
        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        true_values = np.concatenate(true_values)
        pred_values = np.concatenate(pred_values)

        true_classes = true_values.tolist()
        pred_classes = [utils.get_max(pred).index for pred in pred_values]

        end = time.time()
        duration = end - start
        
        logger.info(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, Duration: {duration:.2f}s')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0

            logger.info(f'New best val found! ~ Epoch [{epoch + 1}/{epochs}], Val Loss {avg_val_loss}')

            path = utils.get_path('models', filename='transformer.pth')
            torch.save(model.state_dict(), path)

            checkpoints.update({'best_epoch': epoch+1, 
                                'best_train_loss': avg_train_loss, 
                                'best_val_loss': best_val_loss, 
                                **utils.get_prfs(true=true_classes, pred=pred_classes)})

            if visualize:
                utils.visualize(type='heatmap',
                        values=(true_classes, pred_classes), 
                        labels=('True Values', 'Predicted Values'), 
                        title='Train Heatmap ',
                        classes=classes,
                        coloring=['azure', 'darkblue'],
                        path=utils.get_dir('static', 'transformer'))
                
        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f'Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.')
            break

        scheduler.step()

    cfn = utils.get_path('static', 'transformer', filename='train_checkpoints.json')
    checkpoints.update({'epochs': epoch+1})
    utils.save_json(data=checkpoints, filename=cfn)
    
    if visualize:
        utils.visualize(type='multi-plot',
                        values=[(range(1, len(train_losses) + 1), train_losses), (range(1, len(val_losses) + 1), val_losses)], 
                        labels=('Epoch', 'Loss'), 
                        title='Loss Curves',
                        plot_func=plt.plot,
                        coloring=['brown', 'royalblue'],
                        names=['Training', 'Validation'],
                        path=utils.get_dir('static', 'transformer'))

    logger.info(f'\nTraining complete!\nFinal Training Loss: {avg_train_loss:.6f} & Validation Loss: {best_val_loss:.6f}\n')

def main():
    classes = ['W','R','N1','N2','N3']
    npz_dir = utils.get_dir('data', 'npz')

    samples, chunks = 7680, 32
    seq_len = samples // chunks

    datapaths = split_data(dir=npz_dir, train_size=1, val_size=1, test_size=1)
    
    train_df, val_df, _ = get_dataframes(datapaths, rate=seq_len, exist=True)
    weights = extract_weights(df=train_df, label_col='Consensus')

    datasets = create_datasets(dataframes=(train_df, val_df), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=1, shuffle=[False, False, False], drop_last=False)

    model = Transformer(in_size=9,
                        out_size=len(classes),
                        d_model=64,
                        num_heads=1,
                        num_layers=2,
                        dim_feedforward=2048,
                        dropout=0)
    
    train(data=dataloaders,
          chunks=chunks,
          classes=classes,
          epochs=1,
          patience=30,
          batch_size=512,
          lr=5e-4,
          criterion=utils.WeightedCrossEntropyLoss(weights),
          model=model,
          optimizer='AdamW',
          scheduler=('StepLR', 1.0, 0.98),
          visualize=True)

if __name__ == '__main__':
    main()