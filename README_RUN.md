Running the prediction scripts

This project uses a local virtualenv located at `.venv`.

Preferred run (no system changes):

1. Activate the venv:

   source .venv/bin/activate

2. Install requirements (first time only):

   pip install -r requirements.txt

3. Run the script:

   python predstock.py

Or run without activating the venv:

   .venv/bin/python predstock.py

Helper script:

   ./run_predstock.sh

If you see "Required dependency missing: TensorFlow" when running with your system `python`, use the venv python shown above instead. If you want TensorFlow installed into your system Python, run `python -m pip install tensorflow` for that interpreter (not recommended if `.venv` already works).
