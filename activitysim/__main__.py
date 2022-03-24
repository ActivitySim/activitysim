import sys
import os

def main():
    # clean up message formatting
    if sys.argv and sys.argv[0].endswith('__main__.py'):
        sys.argv[0] = 'activitysim'

    # threadstopper
    if "--fast" not in sys.argv:
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMBA_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

    from .cli.main import main
    main()

if __name__ == '__main__':
    main()
