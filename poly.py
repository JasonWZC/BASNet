"""""
lr is the new learning rate
base_lr is the base learning rate
epoch is the number of iterations
num_epoch is the maximum number of iterations
power controls the shape of the curve (usually greater than 1)
"""""

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

