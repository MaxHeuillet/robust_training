
from losses.trades import trades_loss
from losses.trades_v2 import trades_loss_v2


def get_loss(args, model, x_natural, y, optimizer, train):

    if args.loss_function == 'TRADES':
        return trades_loss(args, model, x_natural, y, optimizer, train)
    elif args.loss_function == 'TRADES_v2':
        return trades_loss_v2(args, model, x_natural, y, optimizer, train)
    # elif args.loss_function == 'MADRY':
    #     return madry_loss(model, x_natural, y, optimizer)
    else:
        print('error - loss not implemented')
