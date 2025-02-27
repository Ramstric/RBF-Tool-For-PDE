import subprocess
import numpy as np

for i in np.arange(0.1, 1, 0.1):
    sigma = i
    # Run script "Wave Eq - String.py" with the argument sigma, which is in the same directory
    subprocess.run(["python", "Wave Eq - String.py", str(sigma)])
    print(f"Finished iteration {i}")

