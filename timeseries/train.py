import matplotlib.pyplot as plt

from . import utils
from .loader import *
from .model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is {device}')

def train(data, classes, epochs, patience, lr, criterion, model, optimizer, scheduler, visualize=False):
    model.to(device)

    train_data, val_data = data

    batches = len(train_data)
    num_classes = len(classes)

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
        model.train()

        total_train_loss = 0.0
        true_values, pred_values = [], []

        for _, (X, y) in enumerate(train_data):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            batch_size, seq_len, _ = y_pred.size()

            y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
            y = y.reshape(batch_size * seq_len)
            
            train_loss = criterion(pred=y_pred, true=y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.detach().cpu().numpy())

        avg_train_loss = total_train_loss / batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X, y in val_data:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)

                batch_size, seq_len, _ = y_pred.size()

                y_pred = y_pred.reshape(batch_size * seq_len, num_classes)
                y = y.reshape(batch_size * seq_len)

                val_loss = criterion(pred=y_pred, true=y)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        true_values = np.concatenate(true_values)
        pred_values = np.concatenate(pred_values)

        true_classes = true_values.tolist()
        pred_classes = [utils.get_max(pred).index for pred in pred_values]
        
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0

            logger.info(f"New best val found! ~ Epoch [{epoch + 1}/{epochs}], Val Loss {avg_val_loss}")

            path = utils.get_path("models", filename="transformer.pth")
            torch.save(model.state_dict(), path)

            checkpoints.update({'best_epoch': epoch+1, 
                                'best_train_loss': avg_train_loss, 
                                'best_val_loss': best_val_loss, 
                                **utils.get_prfs(true=true_classes, pred=pred_classes)})

            if visualize:
                utils.visualize(type="heatmap",
                        values=(true_classes, pred_classes), 
                        labels=("True Values", "Predicted Values"), 
                        title="Train Heatmap ",
                        classes=classes,
                        coloring=['azure', 'darkblue'],
                        path=utils.get_dirs('static'))
                
        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

        scheduler.step()

    cfn = utils.get_path("static", filename="train_checkpoints.json")
    checkpoints.update({'epochs': epoch+1})
    utils.save_json(data=checkpoints, filename=cfn)
    
    if visualize:
        utils.visualize(type="multi-plot",
                        values=[(range(1, len(train_losses) + 1), train_losses), (range(1, len(val_losses) + 1), val_losses)], 
                        labels=("Epoch", "Loss"), 
                        title="Loss Curves",
                        plot_func=plt.plot,
                        coloring=['brown', 'royalblue'],
                        names=["Training", "Validation"],
                        path=utils.get_dirs('static'))

    logger.info(f'\nTraining complete!\nFinal Training Loss: {avg_train_loss:.6f} & Validation Loss: {best_val_loss:.6f}\n')

def main():
    classes = ['W','R','N1','N2','N3']
    npz_dir = utils.get_dir('data', 'npz')
    seq_len = 7680 // 60

    datapaths = split_data(dir=npz_dir, train_size=2, val_size=1, test_size=1)
    
    train_df, val_df, _ = create_dataframes(datapaths, exist=True)
    weights = extract_weights(df=train_df, label_col='Consensus')

    datasets = create_datasets(dataframes=(train_df, val_df), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=256)

    model = Transformer(in_size=3,
                        out_size=len(classes),
                        d_model=64,
                        num_heads=1,
                        num_layers=2,
                        dim_feedforward=2048,
                        dropout=0)
    
    train(data=dataloaders,
          classes=classes,
          epochs=2,
          patience=30,
          lr=5e-4,
          criterion=utils.WeightedCrossEntropyLoss(weights),
          model=model,
          optimizer="AdamW",
          scheduler=("StepLR", 1.0, 0.98),
          visualize=True)

if __name__ == "__main__":
    main()