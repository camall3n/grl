import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from definitions import ROOT_DIR

# %%
data_path = Path(ROOT_DIR, 'results', 'tmaze_mem_interpolation_data.npy')

res_list = np.load(data_path, allow_pickle=True).tolist()

# %%
list_for_df = []
for res in res_list:
    indv_res = {'fuzz': res['fuzz'].item(), 'discrep': res['discrep'].item()}
    list_for_df.append(indv_res)

df = pd.DataFrame(list_for_df)

# %%
sns.lineplot(x='fuzz', y='discrep', data=df)
