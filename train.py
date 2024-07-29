import os
import torch
from model import TwinLite2 as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
from copy import deepcopy
import math
from loss import TotalLoss
import yaml

class ModelEMA:
    
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

def train_net(args, hyp):
    use_ema = args.ema
    if args.seda and args.sell:
        raise ValueError("You can only choose one option.")
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    model = net.TwinLiteNet(args)

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    if not args.is320:
        trainLoader = torch.utils.data.DataLoader(
            myDataLoader.Dataset(hyp, valid=False),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        valLoader = torch.utils.data.DataLoader(
            myDataLoader.Dataset(hyp, valid=True),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        trainLoader = torch.utils.data.DataLoader(
            myDataLoader.Dataset320(hyp, valid=False),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        valLoader = torch.utils.data.DataLoader(
            myDataLoader.Dataset320(hyp, valid=True),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    if args.seda:
        from loss import TotalLossDA
        criteria = TotalLossDA(hyp, is320=args.is320)
    elif args.sell:
        from loss import TotalLossLL
        criteria = TotalLossLL(hyp, is320=args.is320)
    else:
        criteria = TotalLoss(hyp, is320=args.is320)

    start_epoch = 0
    lr = hyp['lr']

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyp['lr'], betas=(hyp['momentum'], 0.999), eps=hyp['eps'], weight_decay=hyp['weight_decay'])
    if use_ema:
        ema = ModelEMA(model)
    if args.resume:
        if os.path.isfile(args.resume):
            if args.resume.split(".")[-1] == "tar":
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                ema.ema.load_state_dict(checkpoint['ema_state_dict'])
                ema.updates=checkpoint['updates']
                print(ema.updates)
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        poly_lr_scheduler(args,hyp,optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

        # train for one epoch
        model.train()
        ema = train(args, trainLoader, model, criteria, optimizer, epoch,scaler,args.verbose,ema if use_ema else None)
        
        
        # validation
        if epoch > 30:
            model.eval()
            da_segment_results,ll_segment_results = val(valLoader, ema.ema if use_ema else model,is320=args.is320,args=args) #da_mIoU_seg, ll_IoU_seg
            msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                        'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
                            da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                            ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
            print(msg)
        torch.save(model.state_dict(), model_file_name)
        torch.save(ema.ema.state_dict(), model_file_name.replace(".pth","_ema.pth"))
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema.ema.state_dict(),
            'updates': ema.updates,
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--savedir', default='./test_full_', help='directory to save the results')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='hyperparameters path')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--type', default="nano", help='')
    parser.add_argument('--is320', action='store_true')
    parser.add_argument('--seda', action='store_true', help='sigle encoder for Drivable Segmentation')
    parser.add_argument('--sell', action='store_true', help='sigle encoder for Lane Segmentation')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--ema', action='store_true', help='')
    args = parser.parse_args()
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    train_net(args, hyp.copy())
