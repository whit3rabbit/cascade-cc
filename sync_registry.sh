#!/bin/bash

# Configuration
BOOTSTRAP_DIR="./cascade_graph_analysis/bootstrap"
PYTHON_BIN="./ml/venv/bin/python3"

# Check if venv exists, fallback to system python or auto-discovery
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(fs.existsSync(path.join(__dirname, '../ml/venv/bin/python3')) ? path.join(__dirname, '../ml/venv/bin/python3') : 'python3')
    # Actually, let's just use what vectorize.py uses or simple python3
    PYTHON_BIN="python3"
fi

echo "[*] Starting Registry Sync (Full Re-vectorization)..."

# Get all library directories in bootstrap
if [ ! -d "$BOOTSTRAP_DIR" ]; then
    echo "[!] Error: Bootstrap directory not found at $BOOTSTRAP_DIR"
    exit 1
fi

LIBS=$(ls -d $BOOTSTRAP_DIR/*/)

for lib in $LIBS; do
    echo ""
    echo "[*] Vectorizing: $(basename $lib)"
    $PYTHON_BIN ml/vectorize.py "$lib" --device cpu
done

echo ""
echo "[*] Updating logic_registry.json..."
node src/update_registry_from_bootstrap.js

echo "[+] Registry Sync Complete!"
