#!/usr/bin/env python3
"""Run the compiled Questline interface and local engine in one Python process."""
import argparse
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5173)
    parser.add_argument('--database', type=Path)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a local port from 1024 to 65535.')
    if sys.version_info < (3, 10):
        parser.error('Questline requires Python 3.10 or newer.')
    if sys.platform != 'darwin':
        parser.error('The restricted C++ execution engine currently supports macOS only.')
    if not (ROOT / 'dist/client/index.html').is_file():
        parser.error('The compiled interface is missing. Run npm ci and npm run build in this folder.')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:
            probe.bind(('127.0.0.1', args.port))
        except OSError:
            parser.error(f'Port {args.port} is already in use. Questline may already be open at http://127.0.0.1:{args.port}.')
    os.chdir(ROOT)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    print(f'Questline · http://127.0.0.1:{args.port}', flush=True)
    print('Checking the restricted C++ compiler. Keep this window open; press Ctrl+C to stop.', flush=True)
    sys.argv = ['questline', '--port', str(args.port)]
    if args.database:
        sys.argv += ['--database', str(args.database.resolve())]
    from backend.server import main as serve
    serve()


if __name__ == '__main__':
    main()
