import torch
import torch.nn as nn
import os
import cv2
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint_sequential

import layers
from progressbar import bar
from LFW_label_utils import logits2trainId, trainId2color, trainId2LabelId

WARNING = lambda x: print('\033[1;31;2mWARNING: ' + x + '\033[0m')
LOG = lambda x: print('\033[0;31;2m' + x + '\033[0m')


# create model
class MobileNetv2_DeepLabv3(nn.Module):
    """
    A Convolutional Neural Network with MobileNet v2 backbone and DeepLab v3 head
        used for Semantic Segmentation on Cityscapes dataset
    """

    """######################"""
    """# Model Construction #"""
    """######################"""

    def __init__(self, params, datasets):
        super(MobileNetv2_DeepLabv3, self).__init__()
        self.params = params
        self.datasets = datasets
        self.pb = bar()  # hand-made progressbar
        self.epoch = 0
        self.init_epoch = 0
        self.ckpt_flag = False
        self.train_loss = []
        self.val_loss = []

        logdir = self.params.summary_dir
        print(f"Logging saved at {logdir}")
        self.summary_writer = SummaryWriter(log_dir=logdir)

        # build network
        mobilenet_block1 = []

        # conv layer 1
        mobilenet_block1.append(nn.Sequential(nn.Conv2d(3, self.params.c[0], 3, stride=self.params.s[0], padding=1, bias=False),
                                   nn.BatchNorm2d(self.params.c[0]),
                                   # nn.Dropout2d(self.params.dropout_prob, inplace=True),
                                   nn.ReLU6()))

        # Inv.res blocks 1-2
        for i in range(2):
            mobilenet_block1.extend(layers.get_inverted_residual_block_arr(self.params.c[i], self.params.c[i + 1],
                                                                t=self.params.t[i + 1], s=self.params.s[i + 1],
                                                                n=self.params.n[i + 1]))

        self.mobilenet1 = nn.Sequential(*mobilenet_block1).cuda()
        self.lowlevel = nn.Conv2d(self.params.c[2], 64, 1).cuda()

        mobilenet_block2 = []

        #Inv. res blocks 3-7
        for i in range(2, 6):
            mobilenet_block2.extend(layers.get_inverted_residual_block_arr(self.params.c[i], self.params.c[i + 1],
                                                                t=self.params.t[i + 1], s=self.params.s[i + 1],
                                                                n=self.params.n[i + 1]))

        # dilated conv layer 1-4
        # first dilation=rate, follows dilation=multi_grid*rate
        rate = self.params.down_sample_rate // self.params.output_stride
        mobilenet_block2.append(layers.InvertedResidual(self.params.c[6], self.params.c[6],
                                             t=self.params.t[6], s=1, dilation=rate))
        for i in range(3):
            mobilenet_block2.append(layers.InvertedResidual(self.params.c[6], self.params.c[6],
                                                 t=self.params.t[6], s=1, dilation=rate * self.params.multi_grid[i]))

        self.mobilenet2 = nn.Sequential(*mobilenet_block2).cuda()

        self.deeplab_context = nn.Sequential(
            layers.ASPP_plus(self.params),
            nn.Conv2d(256, 64, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        ).cuda()

        self.deeplab_cat = nn.Sequential(
            nn.Conv2d(128, self.params.num_class, (3, 3), padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        ).cuda()

        # build loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        # optimizer
        self.opt = torch.optim.RMSprop(list(self.mobilenet1.parameters())+list(self.mobilenet2.parameters())+list(self.deeplab_context.parameters())+list(self.lowlevel.parameters())+list(self.deeplab_cat.parameters()),
                                       lr=self.params.base_lr,
                                       momentum=self.params.momentum,
                                       weight_decay=self.params.weight_decay)

        # initialize
        self.initialize()

        # load data
        self.load_checkpoint()
        self.load_model()

    def forward(self, input):
        mobilenet_part = self.mobilenet1(input)
        lowlevel = self.lowlevel(mobilenet_part)
        mobilenet_full = self.mobilenet2(mobilenet_part)
        context = self.deeplab_context(mobilenet_full)
        cat = torch.cat([context, lowlevel], dim=1)
        return self.deeplab_cat(cat)

    """######################"""
    """# Train and Validate #"""
    """######################"""

    def train_one_epoch(self):
        """
        Train network in one epoch
        """
        print('Training......')

        # set mode train
        self.mobilenet1.train()
        self.mobilenet2.train()
        self.deeplab_cat.train()
        self.lowlevel.train()
        self.deeplab_context.train()

        # prepare data
        train_loss = 0
        train_loader = DataLoader(self.datasets['train'],
                                  batch_size=self.params.train_batch,
                                  shuffle=self.params.shuffle,
                                  drop_last=True,
                                  num_workers=self.params.dataloader_workers)
        train_size = len(self.datasets['train'])
        if train_size % self.params.train_batch != 0:
            total_batch = train_size // self.params.train_batch + 1
        else:
            total_batch = train_size // self.params.train_batch

        # train through dataset
        for batch_idx, batch in enumerate(train_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image.cuda(), label.cuda()

            # checkpoint split
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split, image_cuda)
            else:
                out = self.forward(image_cuda)
            loss = self.loss_fn(out, label_cuda)

            # optimize
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # accumulate
            train_loss += loss.item()

            # record first loss
            if self.train_loss == []:
                self.train_loss.append(train_loss)
                self.summary_writer.add_scalar('loss/train_loss', train_loss, 0)

        self.pb.close()
        train_loss /= total_batch
        self.train_loss.append(train_loss)

        # add to summary
        self.summary_writer.add_scalar('loss/train_loss', train_loss, self.epoch)

    def val_one_epoch(self):
        """
        Validate network in one epoch every m training epochs,
            m is defined in params.val_every
        """
        # TODO: add IoU compute function
        print('Validating:')

        # set mode eval
        self.mobilenet1.eval()
        self.mobilenet2.eval()
        self.deeplab_cat.eval()
        self.lowlevel.eval()
        self.deeplab_context.eval()

        # prepare data
        val_loss = 0
        val_loader = DataLoader(self.datasets['val'],
                                batch_size=self.params.val_batch,
                                shuffle=self.params.shuffle,
                                drop_last=True,
                                num_workers=self.params.dataloader_workers)
        val_size = len(self.datasets['val'])
        if val_size % self.params.val_batch != 0:
            total_batch = val_size // self.params.val_batch + 1
        else:
            total_batch = val_size // self.params.val_batch

        # validate through dataset
        for batch_idx, batch in enumerate(val_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image.cuda(), label.cuda()

            # checkpoint split
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split, image_cuda)
            else:
                out = self.forward(image_cuda)

            loss = self.loss_fn(out, label_cuda)

            val_loss += loss.item()

            # record first loss
            if self.val_loss == []:
                self.val_loss.append(val_loss)
                self.summary_writer.add_scalar('loss/val_loss', val_loss, 0)

        self.pb.close()
        val_loss /= total_batch
        self.val_loss.append(val_loss)

        # add to summary
        self.summary_writer.add_scalar('loss/val_loss', val_loss, self.epoch)

    def Train(self):
        """
        Train network in n epochs, n is defined in params.num_epoch
        """
        self.init_epoch = self.epoch
        if self.epoch >= self.params.num_epoch:
            WARNING('Num_epoch should be smaller than the current epoch. Skipping training......\n')
        else:
            for _ in range(self.epoch, self.params.num_epoch):
                self.epoch += 1
                print('-' * 20 + 'Epoch.' + str(self.epoch) + '-' * 20)

                # train one epoch
                self.train_one_epoch()

                # should display
                if self.epoch % self.params.display == 0:
                    print('\tTrain loss: %.4f' % self.train_loss[-1])

                # should save
                if self.params.should_save:
                    if self.epoch % self.params.save_every == 0:
                        self.save_checkpoint()

                # test every params.test_every epoch
                if self.params.should_val:
                    if self.epoch % self.params.val_every == 0:
                        self.val_one_epoch()
                        print('\tVal loss: %.4f' % self.val_loss[-1])

                # adjust learning rate
                self.adjust_lr()

            # save the last network state
            if self.params.should_save:
                self.save_checkpoint()

            # train visualization
            # self.plot_curve()

    def Test(self):
        """
        Test network on test set
        """
        print('Testing:')
        # set mode eval
        torch.cuda.empty_cache()
        self.mobilenet1.eval()
        self.mobilenet2.eval()
        self.deeplab_cat.eval()
        self.lowlevel.eval()
        self.deeplab_context.eval()

        # prepare test data
        test_loader = DataLoader(self.datasets['test'],
                                 batch_size=self.params.test_batch,
                                 shuffle=False, num_workers=self.params.dataloader_workers)
        test_size = len(self.datasets['test'])
        if test_size % self.params.test_batch != 0:
            total_batch = test_size // self.params.test_batch + 1
        else:
            total_batch = test_size // self.params.test_batch

        # test for one epoch
        for batch_idx, batch in enumerate(test_loader):
            self.pb.click(batch_idx, total_batch)
            image, label, name = batch['image'], batch['label'], batch['label_name']
            image_cuda, label_cuda = image.cuda(), label.cuda()
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split, image_cuda)
            else:
                out = self.forward(image_cuda)

            for i in range(self.params.test_batch):
                idx = batch_idx * self.params.test_batch + i
                id_map = logits2trainId(out[i, ...])
                color_map = trainId2color(self.params.logdir, id_map, name=name[i], save=False)
                #trainId2LabelId(self.params.logdir, id_map, name=name[i])
                image_orig = image[i].numpy().transpose(1, 2, 0)
                image_orig = image_orig * 255
                image_orig = image_orig.astype(np.uint8)

                image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
                color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))

                self.summary_writer.add_image('test/img_%d/orig' % idx, image_orig, idx)
                self.summary_writer.add_image('test/img_%d/seg' % idx, color_map, idx)

    """##########################"""
    """# Model Save and Restore #"""
    """##########################"""

    def save_checkpoint(self):
        save_dict = {'epoch': self.epoch,
                     'train_loss': self.train_loss,
                     'val_loss': self.val_loss,
                     'state_dict': self.state_dict(),
                     'optimizer': self.opt.state_dict()}
        torch.save(save_dict, self.params.ckpt_dir + 'Checkpoint_epoch_%d.pth.tar' % self.epoch)
        print('Checkpoint saved')

    def load_checkpoint(self):
        """
        Load checkpoint from given path
        """
        if self.params.resume_from is not None and os.path.exists(self.params.resume_from):
            try:
                LOG('Loading Checkpoint at %s' % self.params.resume_from)
                ckpt = torch.load(self.params.resume_from)
                self.epoch = ckpt['epoch']
                try:
                    self.train_loss = ckpt['train_loss']
                    self.val_loss = ckpt['val_loss']
                except:
                    self.train_loss = []
                    self.val_loss = []
                self.load_state_dict(ckpt['state_dict'])
                self.opt.load_state_dict(ckpt['optimizer'])
                LOG('Checkpoint Loaded!')
                LOG('Current Epoch: %d' % self.epoch)
                self.ckpt_flag = True
            except:
                WARNING(
                    'Cannot load checkpoint from %s. Start loading pre-trained model......' % self.params.resume_from)
        else:
            WARNING('Checkpoint doesn\'t exist. Start loading pre-trained model......')

    def load_model(self):
        """
        Load ImageNet pre-trained model into MobileNetv2 backbone, only happen when
            no checkpoint is loaded
        """
        if self.ckpt_flag:
            LOG('Skip Loading Pre-trained Model......')
        else:
            if self.params.pre_trained_from is not None and os.path.exists(self.params.pre_trained_from):
                try:
                    LOG('Loading Pre-trained Model at %s' % self.params.pre_trained_from)
                    pretrain = torch.load(self.params.pre_trained_from)
                    self.load_state_dict(pretrain)
                    LOG('Pre-trained Model Loaded!')
                except:
                    WARNING('Cannot load pre-trained model. Start training......')
            else:
                WARNING('Pre-trained model doesn\'t exist. Start training......')

    """#############"""
    """# Utilities #"""
    """#############"""

    def initialize(self):
        """
        Initializes the model parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def adjust_lr(self):
        """
        Adjust learning rate at each epoch
        """
        learning_rate = self.params.base_lr * (1 - float(self.epoch) / self.params.num_epoch) ** self.params.power
        for param_group in self.opt.param_groups:
            param_group['lr'] = learning_rate
        print('Change learning rate into %f' % (learning_rate))
        self.summary_writer.add_scalar('learning_rate', learning_rate, self.epoch)

    def plot_curve(self):
        """
        Plot train/val loss curve
        """
        x1 = np.arange(self.init_epoch, self.params.num_epoch + 1, dtype=np.int).tolist()
        x2 = np.linspace(self.init_epoch, self.epoch,
                         num=(self.epoch - self.init_epoch) // self.params.val_every + 1, dtype=np.int64)
        plt.plot(x1, self.train_loss, label='train_loss')
        plt.plot(x2, self.val_loss, label='val_loss')
        plt.legend(loc='best')
        plt.title('Train/Val loss')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

# """ TEST """
# if __name__ == '__main__':
#     params = CIFAR100_params()
#     params.dataset_root = '/home/ubuntu/cifar100'
#     net = MobileNetv2(params)
#     net.save_checkpoint()
