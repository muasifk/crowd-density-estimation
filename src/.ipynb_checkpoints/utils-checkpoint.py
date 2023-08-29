
import os, random, time
import numpy as np
import torch




def seed_everything(seed):
    ''' SEED Everything '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True 

def calc_game(output, target, L=0):
    ''' Grid Mean Absolute Error (GAME) '''
    output = output[0][0]
    target = target[0]
    H, W = target.shape
    ratio = H / output.shape[0]
    output = cv2.resize(output, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio*ratio)

    assert output.shape == target.shape

    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = output[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = target[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]

            abs_error += abs(output_block.sum()-target_block.sum().float())
            square_error += (output_block.sum()-target_block.sum().float()).pow(2)

    return abs_error # , square_error





