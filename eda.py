"""
Exploratory Data Analysis script for the Amazon ML Challenge.
This is a lightweight wrapper that loads and executes the original notebook
`eda_completed.ipynb` if you have Jupyter installed, or simply prints a
reminder to open the notebook.
"""

import sys
import subprocess
import pathlib

NOTEBOOK = pathlib.Path(__file__).with_name('eda_completed.ipynb')

def main():
    if NOTEBOOK.is_file():
        print(f"Running EDA notebook: {NOTEBOOK}")
        # Convert notebook to script and execute (requires nbconvert)
        try:
            subprocess.run([
                sys.executable, '-m', 'nbconvert', '--to', 'script', '--execute',
                str(NOTEBOOK)
            ], check=True)
        except Exception as e:
            print(f"Failed to execute notebook: {e}")
            print("You can open the notebook manually in Jupyter.")
    else:
        print("EDA notebook not found. Open `eda_completed.ipynb` in Jupyter to explore the data.")

if __name__ == "__main__":
    main()
