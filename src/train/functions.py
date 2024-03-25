import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, config):
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_losses, label='Training Loss')

    ax.plot(epochs, val_losses, label='Validation Loss')

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training and Validation Losses', fontsize=16)
    ax.legend(fontsize=12)

    figure_path = f"{config.model_name}_loss.png"
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    # plt.show()


def lanseloss(y_pred, action_pred, y_true, action_true, space=None, space_truth=None):
    # y_true = y_true.type(torch.LongTensor).to(y_true.device)

    # pos_loss = torch.sqrt(F.mse_loss(y_pred, y_true))  # RMSE
    # pos_loss = F.mse_loss(y_pred, y_true)  # MSE
    # pos_loss = F.smooth_l1_loss(y_pred[:, :3], y_true[:, :3], beta=1)
    # pos_loss = pos_loss + F.smooth_l1_loss(y_pred[:, 3:], y_true[:, 3:], beta=0.1)
    pos_loss = F.smooth_l1_loss(y_pred, y_true, beta=0.1)
    pos_loss_mae = F.l1_loss(y_pred, y_true)

    action_loss = F.binary_cross_entropy(action_pred.squeeze(), action_true.squeeze())
    # space_loss = 0
    # if space is not None:
    #     space_loss = F.mse_loss(space, space_truth)

    return pos_loss + action_loss, pos_loss, pos_loss_mae  # pos_loss_mae


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1
        scale = self.d_model ** -0.5 * min(current_step ** -0.5, current_step * self.warmup_steps ** -1.5)
        return [base_lr * scale for base_lr in self.base_lrs]
