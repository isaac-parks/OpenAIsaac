import os
import sys
from utils import write_raw_content, init_dataset
from mlp_train import main as mlp_train
from mlp_run import main as mlp_run

def parse_args():
    # super scuffed rn god i love python
    allowed_args = ['--compare']
    allowed_args.sort()
    sys.argv = sys.argv[1:]
    args = sys.argv
    args.sort()
    arg_list = []
    for i in range(len(args)):
        if args[i] == allowed_args[i]:
            arg_list.append(allowed_args[i])

    return {
        'compare': True if '--compare' in arg_list else False
    }

def main():
    # args = parse_args()
    compare = parse_args().get('compare')
    ds_fname = 'dataset'
    w_fname = 'weights.pkl'
    if not os.path.isfile(ds_fname):
        write_raw_content(ds_fname)

    words, stoi, itos = init_dataset(ds_fname)
    if not os.path.isfile(w_fname) or (compare and not os.path.isfile('init_weights')):
        mlp_train(words, stoi, w_fname, compare)
    else:
        print(f'\nUsing prexisting weights from file: "{w_fname}"')


    if compare:
        p = '\n'.join(mlp_run(words, itos, 'init_weights').split('.')) 
        print(f'\nUNTRAINED predictions based:')
        print(f'{p}\n')

    
    p = '\n'.join(mlp_run(words, itos, w_fname).split('.')) 
    print(f'\nTrained predictions based on file - "{ds_fname}":')
    print(p)
    
if __name__ == '__main__':
    main()