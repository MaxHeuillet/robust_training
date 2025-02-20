
# from old_files.trades import trades_loss
from losses.trades import trades_loss
from losses.classic_at import classic_at_loss
# from losses.trades_v3 import trades_loss_v3
from losses.eval_trades import trades_loss_eval

from losses.eval_classic_at import classic_at_loss_eval

def get_loss(setup, model, x_nat, y, ):

    loss_function = setup.config.loss_function

    if loss_function == 'TRADES_v2':
        # print('returns TRADES_v2 loss')
        return trades_loss(setup, model, x_nat, y, )
    elif loss_function == 'CLASSIC_AT':
        # print('returns APGD loss')
        return classic_at_loss(setup, model, x_nat, y, )
    else:
        print('error - loss not implemented')


def get_eval_loss(setup, model, x_nat, y):

    loss_function = setup.config.loss_function

    if loss_function == 'TRADES_v2':
        # print('returns TRADES_v2 loss')
        return trades_loss_eval(setup, model, x_nat, y, )
    elif loss_function == 'CLASSIC_AT':
        # print('returns APGD loss')
        return classic_at_loss_eval(setup, model, x_nat, y, )

    return trades_loss_eval(setup, model, x_nat, y)

