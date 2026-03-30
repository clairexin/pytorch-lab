import os
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

scripts = ["linear_regression.py", "logistic_regression.py", "mlp.py"]

os.makedirs("data", exist_ok=True)

# Run all training scripts
for script in scripts:
    print(f"\n{'='*50}")
    print(f"Running {script}...")
    print('='*50)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"Error running {script}")
        sys.exit(1)

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
