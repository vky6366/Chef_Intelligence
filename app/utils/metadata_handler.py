# utils/metadata_handler.py
from typing import List, Dict, Any
import pandas as pd

def df_to_docs(df: pd.DataFrame, text_col: str, meta_cols: List[str]) -> List[Dict[str, Any]]:
    docs = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, "") or "")
        meta = {c: row.get(c, None) for c in meta_cols}
        if text.strip():
            docs.append({"text": text, "metadata": meta})
    return docs
