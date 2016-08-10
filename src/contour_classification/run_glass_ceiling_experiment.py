# Executes glass ceiling experiments

import run_experiments as re
import pandas as pd
import numpy as np
meltype = 1
results = re.run_glassceiling_experiment(meltype)
df = pd.DataFrame(results.values(), index=results.keys())
print df.describe()