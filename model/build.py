from .SwinIR import SwinIR
from .SwinPlus import SwinPlus
from .Restormer import Restormer
from .NAFNet import NAFNet, NAFNetLocal
from .Uformer import Uformer


def build_model(args):
    if args.arch == 'restormer':
        model = Restormer(use_checkpoint=args.use_checkpoint)
        
    elif args.arch == 'swinir':
        model = SwinIR(drop_path_rate=args.drop_path_rate, use_mask=args.use_mask, use_checkpoint=args.use_checkpoint)

    elif args.arch == 'swinplus':
        model = SwinPlus(drop_path_rate=args.drop_path_rate, use_mask=args.use_mask, use_checkpoint=args.use_checkpoint)

    elif args.arch == 'uformer':
        model = Uformer(use_checkpoint=args.use_checkpoint)

    elif args.arch == 'naf':
        if args.mode == 'train':
            model = NAFNet(use_checkpoint=args.use_checkpoint)
        else:
            model = NAFNetLocal(train_size=(1, 4, args.train_size, args.train_size))

    else:
        raise ValueError(f'Unimplemented model: {args.arch}')
 
    return model
    