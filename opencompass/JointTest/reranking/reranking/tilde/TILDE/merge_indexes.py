# _*_ coding:utf-8 _*_
#**********************************************************************************************************************
# author:       Ahmed Mourad
# date:         08 Dec 2021
# description:  Utility script to merge TILDE expanded index given a list of directories
#**********************************************************************************************************************
from argparse import ArgumentParser
import os
import h5py
import numpy as np

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    docids = None
    
    with h5py.File(os.path.join(args.output_dir, "tildev2_index.hdf5"), "w") as h5fw:
        counter = 0
        for index_path in args.input_dirs:
            if os.path.isdir(index_path):
                print(index_path)
                # read the h5py ([tokens], [scores])
                f = h5py.File(os.path.join(index_path, "tildev2_index.hdf5"), 'r')
                # name of the dataset
                dset_name = list(f.keys())[0]
                tokens_scores = f[dset_name][:]
                
                # initial creation of the dataset
                if counter == 0:
                    h5fw.create_dataset(dset_name, data=tokens_scores, maxshape=(None,))
                    docids = np.load(os.path.join(index_path, "docids.npy"))
                else:
                    # resize the dataset
                    h5fw[dset_name].resize((counter+tokens_scores.shape[0],))            
                    h5fw[dset_name][counter:counter+tokens_scores.shape[0]] = tokens_scores[:]

                    docids = np.append(docids, np.load(os.path.join(index_path, "docids.npy")))
                
                counter += tokens_scores.shape[0]
                f.close()

                assert len(docids) == len(h5fw[dset_name])
    
    np.save(os.path.join(args.output_dir, "docids.npy"), docids)

if __name__ == "__main__":
    parser = ArgumentParser(prog="Merge TILDE v2 indexes")
    parser.add_argument('-o', '--output_dir', type=str, default='', help="output directory for the final index")
    parser.add_argument('input_dirs', nargs='+', help="list of directories for different indexes")
    args = parser.parse_args()

    main(args)