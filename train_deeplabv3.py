import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from torch.utils.data import DataLoader
from data.GID_dataset import GIDDataset
from options.train_options import TrainOptions


def main(opts):

    log_path = os.path.join(opts.checkpoints_dir, "logs")
    writer =  SummaryWriter(log_path)
    logger.add(os.path.join(opts.checkpoints_dir, "loss.lg"))

    train_data = GIDDataset()
    train_data.initialize(opt=opts)
    data_size = len(train_data)

    train_data = DataLoader(train_data, batch_size=opts.batchSize, shuffle=True, num_workers=16)
    nc = opts.label_nc + 1 if opts.contain_dontcare_label \
            else opts.label_nc
    classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=nc, aux_loss=None)
    if opts.continue_train:
        checkpoint = torch.load(opts.deeplab_resume)
        classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier.cuda()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(opts.niter):
        for i, da, in enumerate(train_data):
            classifier.train()

            optimizer.zero_grad()
            img, mask = da['image'], da['label']

            img = img.cuda()
            mask = mask.long().cuda()[:,0,:,:]

            y_pred = classifier(img)['out']
            loss = criterion(y_pred, mask)
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                logger.info("{} epoch iteration {} loss {}".format(epoch, i, loss.item()))
                writer.add_scalar('Loss', loss.item(), epoch*data_size+i)

        model_path = os.path.join(opts.checkpoints_dir, 'deeplab_' + str(epoch) + '.pth')
        torch.save({'model_state_dict': classifier.state_dict()}, model_path)
        print('Save to:', model_path)


if __name__ == '__main__':
    opts = TrainOptions().parse()
    opts.checkpoints_dir = os.path.join(opts.deeplab_train_dir, opts.name)
    
    print("Opt", opts)

    path = opts.checkpoints_dir
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    main(opts)
