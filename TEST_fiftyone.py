import os
import fiftyone as fo
import fiftyone.core.session as fos
import socket

# ---------------------
# Force in-memory mode
# ---------------------
#os.environ["FIFTYONE_DISABLE_MONGODB"] = "True"
os.environ["FIFTYONE_LOG_LEVEL"] = "debug"

print("\n=== Step 1: Creating temporary in-memory dataset ===")
dataset = fo.Dataset(name="debug_temp_dataset")
dataset.add_sample(fo.Sample(filepath="/dev/null"))  # dummy sample

print("Dataset created:", dataset.name)
print("Number of samples:", len(dataset))

print("\n=== Step 2: Launching FiftyOne app on a free port ===")
free_port = 5181
while True:
    try:
        sock = socket.socket()
        sock.bind(("localhost", free_port))
        sock.close()
        break
    except OSError:
        free_port += 1

print(f"Launching FiftyOne on port {free_port}...")
session = fo.launch_app(dataset, port=free_port, auto=False)  # don't auto-open browser
print("Session object created.")

print("\n=== Step 3: Checking session status ===")
try:
    print("Session status:", session.status)
except Exception as e:
    print("❌ Failed to check session status:", str(e))

print("\n=== Step 4: Backend info ===")
print("Session URL:", session.url)
print("Dataset name:", session.dataset.name)

print("\n✅ Script completed. Open this in your browser (after SSH tunnel):")
print(f"http://127.0.0.1:{free_port}")
import fiftyone.core.service as fos
print("Mongo binary path:", fos.DatabaseService.find_mongod())