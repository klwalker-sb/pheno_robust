#!/usr/bin/env python
import argparse
from pheno_robust.pheno import make_pheno_vars
from pheno_robust.version import __version__

def main():
       
    def check_for_list(arg_input):
        '''lists passed from Bash will not be parsed correctly. 
           This function converts variables in Bash scripts that are 
           entered as strings "[X,X,X]" into lists in Python.
        '''
        if isinstance(arg_input, list):
            arg_input = arg_input
        elif arg_input.startswith('['):
            arg_input = arg_input[1:-1].split(',')
            try:
                arg_input = list(map(int, arg_input))
            except:
                pass
        return arg_input

    parser = argparse.ArgumentParser(description='scripts to augment info for data exploration in notebooks',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest='process')

    available_processes = [
                           'version',
                           'make_pheno_vars'
                          ]

    for process in available_processes:
        subparser = subparsers.add_parser(process)
        if process == 'version':
            continue
        if process == 'make_pheno_vars':
            subparser.add_argument('--out_dir', dest='out_dir', help='out directory for processed outputs', default=None)
            subparser.add_argument('--start_yr', dest ='start_yr', help='year to map (first if spans two)', default=2010, type=int)
            subparser.add_argument('--start_mo', dest ='start_mo', default=2010, type=int,
                                   help='month to start calendar (e.g 1 if mapping Jan-Dec; 7 if mapping July-Jun')
            subparser.add_argument('--img_dir', dest ='img_dir', help='directory containing images')
            subparser.add_argument('--spec_index', dest='spec_index', help='Spectral index to explore. options are...', default='evi2')
            subparser.add_argument('--pheno_vars', dest ='pheno_vars', help='pheno variables to calculate - in order of band output')
            subparser.add_argument('--sigdif', dest ='sigdif', help='increase/decrease from median considered significant for peak',
                                  default = 500)
            subparser.add_argument('--pad_days', dest ='pad_days', help='number of days to pad on each side of season for curve fitting',
                                  default = '[20,20]')
        
    args = parser.parse_args()

    if args.process == 'version':
      print(__version__)
      return
                                   
        if args.process == 'make_pheno_vars':
        make_pheno_vars(img_dir = args.img_dir,
                        out_dir = args.out_dir,
                        start_yr = args.start_yr,
                        start_mo = args.start_mo,
                        spec_index = args.spec_index, 
                        pheno_vars = check_for_list(args.pheno_vars),
                        sigdif = args.sigdif,
                        pad_days = check_for_list(args.pad_days))
        
if __name__ == '__main__':
    main()
