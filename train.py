import argparse
import os
import torch
from datetime import datetime
from tqdm import tqdm

from datasets import CustomDataset


def get_write_text(prefix, loss):
    s = prefix + ' '
    for k, v in loss.items():
        s += f"{k}: {v:6.4f} | "
    s += '\n'
    return s


def train(opt):
    print('Creating dataset...')
    mydataset = CustomDataset(opt)
    print('Dataset created.')
    opt.classes = mydataset.num_class

    print('Initializing Trainer...')
    model = Trainer(opt)
    print('Trainer initialized.')

    if opt.pretrained:
        model.load()
        
    train_loader = torch.utils.data.DataLoader(
        mydataset, 
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_threads,
        pin_memory=True,
        drop_last=True,
    )

    print('Start training.')
    start_epoch = 0
    best = 1000.
    for epoch in range(start_epoch + 1, opt.epochs + 1):
        desc = f"[{epoch}/{opt.epochs}] "
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, miniters=1)
        for i, data in pbar:
            model.run(data)
            model.compute_loss()
            loss = model.get_loss()
            s = 'Loss: %7.4f ' % (loss['loss'])
            pbar.set_description(desc + s)

        # Save training results
        if loss['loss'] < best:
            best = loss['loss']
            model.save(epoch, best=True)
            print(f"Save best loss model at epoch {epoch}.")
        if epoch % opt.save_epoch == 0:
            model.save(epoch)
            print(f"Save model at epoch {epoch}.")
        model.update_learning_rate(epoch)

        f_log = open(opt.log_path, 'a')
        f_log.write(get_write_text(desc, loss))
        f_log.close()

    if opt.epochs % opt.save_epoch != 0:
        model.save(opt.epochs)
        print(f"Save model at epoch {epoch}.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # System setting
    parser.add_argument('--lincls', action='store_true', help='use data label to train linear classification after unsupervised training')
    parser.add_argument('--gpus_str', type=str, default='0', help='use which graphic card')
    parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
    # Dataset setting
    parser.add_argument('--data_path', type=str, default='data_path.txt', 
                        help='to a txt list of folder or directly to the folder contain two sub-folder: img/ label')
    # Common training setting
    parser.add_argument('--input_size', type=int, default=64, help='input image size')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    # Optimizer setting
    parser.add_argument('--ratio', type=float, default=0.1, help='determine the size of mask for image transformation')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate for adam')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
    parser.add_argument('--pool_size', type=int, default=2, help='the size of maxpooling kernal')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--lambda_NP', type=float, default=1.0, help='weight for MCNP loss')
    # Model setting
    parser.add_argument('--pred_dim', type=int, default=512, help='hidden dimension of the predictor')
    # Pretrained setting
    parser.add_argument('--pretrained', action='store_true', help='if specified, load pretrained weight before training')
    parser.add_argument('--weight_dir', type=str, default='checkpoints', help='folder path of weights')
    parser.add_argument('--load_epoch', type=str, default='0', help='which epoch to load')
    parser.add_argument('--res18_weight', type=str, default='resnet18-5c106cde.pth', help='path to resnet18 imagenet weights')
    # Save setting
    parser.add_argument('--save_dir', type=str, default='train_log', help='the folder path to save training results')
    parser.add_argument('--save_epoch', type=int, default=20, help='periodical save the model')
    opt = parser.parse_args()

    opt.isTrain = True
    opt.device = 'cuda' if len(opt.gpus_str)>0 else 'cpu'
    opt.gpu_ids = opt.gpus_str.split(',') if len(opt.gpus_str)>0 else []
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # opt.epochs = opt.n_epochs + opt.n_epochs_decay
    opt.epochs = opt.n_epochs

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    opt.save_dir_model = os.path.join(opt.save_dir, current_time)
    os.makedirs(opt.save_dir_model)
    opt.log_path = os.path.join(opt.save_dir, current_time, 'loss_log.txt')

    if opt.lincls:
        from trainer_lincls import Trainer
        opt.pretrained = True
    else:
        from trainer import Trainer

    train(opt)
