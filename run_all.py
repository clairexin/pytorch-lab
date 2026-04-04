import argparse
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

LABS = {
    "1-linear": {"script": "1_linear_regression.py", "data": "data/1_linear_regression.json"},
    "2-logistic": {"script": "2_logistic_regression.py", "data": "data/2_logistic_regression.json"},
    "3-mlp": {"script": "3_mlp.py", "data": "data/3_mlp.json"},
    "4-mnist": {"script": "4_mnist.py", "data": "data/4_mnist.json"},
}

parser = argparse.ArgumentParser(description="Run PyTorch lab exercises and view dashboard")
parser.add_argument(
    "labs",
    nargs="*",
    choices=list(LABS.keys()),
    help=f"Labs to run (default: all). Choose from: {', '.join(LABS.keys())}",
)
parser.add_argument("--no-server", action="store_true", help="Skip launching the web server")
parser.add_argument("--force", action="store_true", help="Re-run labs even if data already exists")
args = parser.parse_args()

selected = args.labs or list(LABS.keys())

os.makedirs("data", exist_ok=True)

for name in selected:
    lab = LABS[name]
    if not args.force and os.path.exists(lab["data"]):
        print(f"\n{'='*50}")
        print(f"Skipping {lab['script']} (data already exists: {lab['data']})")
        print('='*50)
        continue
    print(f"\n{'='*50}")
    print(f"Running {lab['script']}...")
    print('='*50)
    result = subprocess.run([sys.executable, lab["script"]])
    if result.returncode != 0:
        print(f"Error running {lab['script']}")
        sys.exit(1)

if args.no_server:
    sys.exit(0)

# Start local server and open browser
PORT = 8000
Handler = SimpleHTTPRequestHandler

print(f"\n{'='*50}")
print(f"Starting server at http://localhost:{PORT}/web/")
print(f"Press Ctrl+C to stop")
print('='*50)

threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{PORT}/web/")).start()

with TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
