import subprocess
import time
import os
from glob import glob
from pathlib import Path
import pandas as pd
from mcr.optimize import analyze_qc_file


def fasttodd_optimization(qc_path):
    work_dir = "./quantum-circuit-optimization"

    # Define the output file path beforehand
    output_filepath = str(Path(f"{work_dir}/circuits/outputs/rrr").resolve())

    # Execute the Rust command asynchronously
    cmd = ["cargo", "+nightly", "run", "--release", qc_path]
    # cmd = ["cargo", "+nightly", "run", "--release", "-r", "FastTODD", abs_tmp_qc_filepath]
    process = subprocess.Popen(
        cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    print(f"Running command: {' '.join(cmd)}")
    print(f"Waiting for output file: {output_filepath}")

    # Wait for the file to be generated (maximum 60 seconds)
    timeout = 60  # seconds
    interval = 0.5  # polling interval
    elapsed = 0

    while not os.path.exists(output_filepath):
        time.sleep(interval)
        elapsed += interval
        if elapsed >= timeout:
            print("Timeout: Output file was not generated")
            break

    # Terminate the Rust process if it's still running after the file is generated
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    # Optionally log output (optional)
    stdout, stderr = process.communicate()
    print("Rust stdout:\n", stdout)
    print("Rust stderr:\n", stderr)

    if os.path.exists(output_filepath):
        ancilla, T = analyze_qc_file(output_filepath)
        os.remove(output_filepath)
        # os.remove(tmp_qc_filepath)
        df = pd.DataFrame({"added_ancilla": [ancilla], "after_opt_T_count": [T]})
        return df
    else:
        return {"error": "Output file was not found"}


def main():
    file = "tmp/1d009ee8-53b5-4d1b-a79c-737f1a4e5d09.qc"
    ancilla_count, t_count = fasttodd_optimization(file)
    print(f"File: {file}, Ancilla Count: {ancilla_count}, T Count: {t_count}")
    # os.remove(file)


if __name__ == "__main__":
    main()
