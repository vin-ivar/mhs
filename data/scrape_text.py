import pandas as pd
import glob

for f in glob.glob('./data/all-processed/*.csv'):
    df = pd.read_csv(f, header=0, sep=',')
    outfile = f"./data/embed/{f.split('/')[-1].split('.')[0]}.txt"
    with open(outfile, 'w') as fout:
        fout.write("\n".join(df.text.astype(str).tolist()))
