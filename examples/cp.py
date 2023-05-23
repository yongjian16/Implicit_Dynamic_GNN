import shutil

import os 


def create_run_file_with_input_seed():
    for fname in os.listdir('.'):
        if len(fname) < 6: continue
        newfile = fname.replace('_56', '')
        # shutil.copyfile(fname, newfile)
        with open(fname, 'rt') as f:
            text = ''.join(f.readlines())
            text = text.replace('seed 56', 'seed $1')
            text = text.replace('epoch 1', 'epoch 200')
            text = text.replace('device cpu', 'device cuda')
        
        with open(newfile, 'wt') as f:
            f.write(text)

def print_methods():
    for fname in os.listdir('examples/task-train-brain10/brain10-none/'):
        if len(fname) < 6: continue
        print(fname.split('~')[3].split('_')[1])

print_methods()