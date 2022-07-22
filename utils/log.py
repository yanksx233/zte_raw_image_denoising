import os
import shutil
import logging


"""
    - ./log
        - save_dir1
            - train
                - args.log
                - train.log
                - epoch.log
                - checkpoint
            - test
                - test.log
                - result
        - save_dir2
            ...
        - save_dir 3
            ...
        ...

"""


def get_logger(args):
    if os.path.exists(os.path.join(args.log_dir, args.save_dir, args.mode)):
        if args.mode != 'train':
            shutil.rmtree(os.path.join(args.log_dir, args.save_dir, args.mode))
            os.makedirs(os.path.join(args.log_dir, args.save_dir, args.mode, 'result'))
    else:
        os.makedirs(os.path.join(args.log_dir, args.save_dir, args.mode, 'checkpoint' if args.mode == 'train' else 'result'))

    # save args
    args_file = open(os.path.join(args.log_dir, args.save_dir, args.mode, 'args.log'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(20) + '\t' + str(v) + '\n')

    # logger setting
    logger = logging.getLogger(name=args.mode + 'logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)s] - %(message)s')

    file_handler = logging.FileHandler(os.path.join(args.log_dir, args.save_dir, args.mode, args.mode + '.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if args.mode == 'train':
        epoch_handler = logging.FileHandler(os.path.join(args.log_dir, args.save_dir, args.mode, 'epoch.log'))
        epoch_handler.setLevel(logging.INFO)
        epoch_handler.setFormatter(formatter)
        logger.addHandler(epoch_handler)

    return logger


if __name__ == '__main__':
    pass
    # from option import args
    #
    # args.save_dir = '123'
    # args.mode = 'test'
    #
    # logger = get_logger(args)
    # logger.info('info')
    # logger.debug('debug')
    class args:
        log_dir = './log' 
        mode = 'test'
        save_dir = '123'
    get_logger(args)

