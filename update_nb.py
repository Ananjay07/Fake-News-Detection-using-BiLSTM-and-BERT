import nbformat

notebook_path = 'd:/LPU/Sem6/NLP/NLP.DL.Prjct/notebooks/02_bilstm_attention.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

new_source = """from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("../outputs")

# Load LIAR dataset
train_df = pd.read_csv(OUTPUT_DIR / "train_processed.csv")
test_df  = pd.read_csv(OUTPUT_DIR / "test_processed.csv")
liar_df = pd.concat([train_df, test_df], ignore_index=True)

# Load ISOT dataset
fake_df = pd.read_csv("../data/ISOT/Fake.csv")
true_df = pd.read_csv("../data/ISOT/True.csv")
fake_df["label"] = 0
true_df["label"] = 1
fake_df["text"] = fake_df["title"] + " " + fake_df["text"]
true_df["text"] = true_df["title"] + " " + true_df["text"]
fake_df = fake_df[["text", "label"]]
true_df = true_df[["text", "label"]]
isot_df = pd.concat([fake_df, true_df], ignore_index=True)

# Reduce ISOT dataset size for faster CPU training
isot_df = isot_df.sample(12000, random_state=42).reset_index(drop=True)

# Combine datasets
combined_df = pd.concat([liar_df, isot_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and test
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
train_df.head()"""

# Find the cell that loads the data and replace its source
for cell in nb.cells:
    if cell.cell_type == 'code' and 'train_df = pd.read_csv(OUTPUT_DIR / "train_processed.csv")' in cell.source:
        cell.source = new_source
        break

# Clear the outputs of all cells since the data changed
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook updated successfully.")
