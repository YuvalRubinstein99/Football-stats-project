"""Project entry point: train, report, check, app, or predict."""
from pathlib import Path
import argparse
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'report', 'check', 'app', 'predict'])
    args, remaining = parser.parse_known_args()
    command = args.command
    sys.path.insert(0, str(ROOT / 'src'))
    if command in ('app', 'predict'):
        if command == 'app':
            from prediction_app import main as live_main
        else:
            from live_predictions import main as live_main
        live_main(remaining)
        return
    if remaining:
        parser.error('unrecognized arguments: ' + ' '.join(remaining))
    if command == 'train':
        from prediction_bivariate import main as train
        train()
        runpy.run_path(str(ROOT / 'src/build_bivariate_notebook.py'), run_name='__main__')
    elif command == 'report':
        runpy.run_path(str(ROOT / 'src/build_bivariate_notebook.py'), run_name='__main__')
    else:
        runpy.run_path(str(ROOT / 'scripts/verify_project.py'), run_name='__main__')


if __name__ == '__main__':
    main()
