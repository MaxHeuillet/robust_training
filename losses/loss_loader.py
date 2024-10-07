
from losses.trades import trades_loss
from losses.trades_v2 import trades_loss_v2
from losses.apgd import apgd_loss
# from losses.trades_v3 import trades_loss_v3
from losses.eval_trades import trades_loss_eval

def get_loss(args, model, x_natural, y, optimizer):

    if args.loss_function == 'TRADES':
        # print('returns TRADES loss')
        return trades_loss(args, model, x_natural, y, optimizer)
    elif args.loss_function == 'TRADES_v2':
        # print('returns TRADES_v2 loss')
        return trades_loss_v2(args, model, x_natural, y, optimizer)
    elif args.loss_function == 'APGD':
        # print('returns APGD loss')
        return apgd_loss(args, model, x_natural, y, optimizer)
    # elif args.loss_function == 'TRADES_v3':
    #     return trades_loss_v3(args, model, x_natural, y, optimizer)
    # elif args.loss_function == 'MADRY':
    #     return madry_loss(model, x_natural, y, optimizer)
    else:
        print('error - loss not implemented')


def get_eval_loss(args, model, x_natural, y):

    return trades_loss_eval(args, model, x_natural, y)


