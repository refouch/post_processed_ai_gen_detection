"""
Quick script to tailor the dataset to our current needs:
    - Reduce it greatly to avoid big computztion times
    - Perform a new train/val/test split across all 3 categories
    - /!\ Keep class balanced across all sets 

MAIN IDEA: 
1. log all file names in the dataset
2. Add features by parsing the file path, which contains all information needed
3. Pivot the table to get only one file ID per row, pointing to the three different transforms using the filepaths
-> Important because we dont want the model to see the same image even altered in train and in train
4. Performed a balanced train/val/test split
"""

import os, re, glob
import pandas as pd
from sklearn.model_selection import train_test_split
 
DIR = "RRDataset_test/RRDataset_final"
TRANSFORMS = ["original", "transfer", "redigital"]
CLASSES = ["ai", "real"]
SEED = 42
 
### I. Building LONG dataframe: 1 row per file path
paths = []
for t in TRANSFORMS:
    for c in CLASSES:
        paths += glob.glob(os.path.join(DIR, t, c, "*"))
long = pd.DataFrame({"filepath": paths})

def parse(p):
    fn = os.path.splitext(os.path.basename(p))[0]
    cls = os.path.basename(os.path.dirname(p))
    transform = os.path.basename(os.path.dirname(os.path.dirname(p)))

    # Deleting class and transform tokens to avoid duplicates
    tokens = [t for t in fn.split("_") if t not in TRANSFORMS and t != "real"]
    fn = "_".join(tokens)

    m = re.match(r"^(.*?)_?(\d+)$", fn)
    if m:
        stem, img_id = m.group(1), str(int(m.group(2)))  
    else:
        stem, img_id = fn, ""

    if cls == "real":
        context = "None"
        uid = f"real_{img_id}"         
    else:
        context = stem
        uid = f"ai_{context}_{img_id}" 
    return cls, transform, context, uid
 
long[["label","transform","context","unique_id"]] = long["filepath"].apply(lambda p: pd.Series(parse(p)))


### II. Build WIDE dataframe with pivot: 1 row per each unique photo ID, regardless of transform
wide = long.pivot_table(index=["unique_id","label","context"],
                        columns="transform", values="filepath",
                        aggfunc="first").reset_index()
wide.columns.name = None
before = len(wide)

wide = wide.dropna(subset=TRANSFORMS).reset_index(drop=True)


### III. Perform a BALANCED split across all labels AND 

wide["strat"] = wide["label"] + "_" + wide["context"].fillna("none") # grouping label and context for stratify

train, temp = train_test_split(wide, test_size=0.30, random_state=SEED, stratify=wide["strat"])
val, test   = train_test_split(temp, test_size=0.50, random_state=SEED, stratify=temp["strat"])

for name, part in [("train",train),("val",val),("test",test)]: # Adding the split in the datasep
    wide.loc[part.index, "split"] = name
 
for t in TRANSFORMS:
    wide[t] = "dataset/" + wide[t] # Adding back the absolute path from root

wide.drop(columns="strat").to_csv("rrdataset_new_split.csv", index=False) # Exporting dataset
 
### SANITY CHECK
print(f"\nwide: {len(wide)} unique photos | {len(wide)* 3} individual files \n")
for s in ["train", "val", "test"]:
    sub = wide[wide.split == s]
    n_files = len(sub) * 3
    print(f"=== {s} ({n_files} individual files / {len(sub)} unique photos ===")
    print("  real/fake:", (sub.label.value_counts() * 3).to_dict())
    print("  transform:", {t: len(sub) for t in TRANSFORMS})
    print("  context (ai):", sub.loc[sub["label"] == "ai", "context"].nunique(), "classes")

print("\nAll detected contexts", sorted(wide.loc[wide.label == "ai", "context"].unique()))
# Dataset is balanced !!
