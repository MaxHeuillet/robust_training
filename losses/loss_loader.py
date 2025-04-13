
# from old_files.trades import trades_loss
from losses.cross_entropy import ce_loss
from losses.trades import trades_loss
from losses.classic_at import classic_at_loss

# from losses.trades_v3 import trades_loss_v3
from losses.eval_trades import trades_loss_eval

from losses.eval_classic_at import classic_at_loss_eval

def get_loss(config, model, x_nat, y, ):

    loss_function = config.loss_function

    if loss_function == 'TRADES_v2':
        # print('returns TRADES_v2 loss')
        return trades_loss(config, model, x_nat, y, )
    elif loss_function == 'CLASSIC_AT':
        # print('returns APGD loss')
        return classic_at_loss(config, model, x_nat, y, )
    elif loss_function == 'CE':
        # print('returns CE loss')
        return ce_loss(model, x_nat, y, )
    else:
        print('error - loss not implemented')


def get_eval_loss(config, model, x_nat, y):

    loss_function = config.loss_function

    if loss_function == 'TRADES_v2':
        # print('returns TRADES_v2 loss')
        return trades_loss_eval(config, model, x_nat, y, )
    elif loss_function == 'CLASSIC_AT':
        # print('returns APGD loss')
        return classic_at_loss_eval(config, model, x_nat, y, )
    elif loss_function == 'CE':
        # print('returns CE loss')
        losses, logits_nat =  ce_loss(model, x_nat, y, )
        return losses, logits_nat, logits_nat
    else:
        print('not implemented error')

