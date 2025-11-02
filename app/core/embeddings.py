# embedding_manager.py
from typing import Iterable, List, Dict, Any, Union
from uuid import uuid4
import pandas as pd

from sentence_transformers import SentenceTransformer
from qdrant_client import models

from utils.qdrant_client import qclient, ensure_collection   # function import OK now
from utils.metadata_handler import df_to_docs

class EmbeddingManager:
    def __init__(self, collection: str, emb: Union[str, SentenceTransformer], df: pd.DataFrame, TEXT_COL: str, META_COLS: List[str],):
        self.collection = collection
        self.emb = SentenceTransformer(emb) if isinstance(emb, str) else emb
        self.df = df.fillna("")
        self.TEXT_COL = TEXT_COL
        self.META_COLS = META_COLS

    @staticmethod
    def batched(seq: Iterable, size: int):
        buf = []
        for x in seq:
            buf.append(x)
            if len(buf) == size:
                yield buf
                buf = []
        if buf:
            yield buf

    def upsert_docs(self, docs: List[Dict[str, Any]], batch_size: int = 16, wait: bool = False) -> int:
        docs = [d for d in docs if d.get("text")]
        if not docs:
            return 0

        # infer embedding dimension and ensure collection
        dim = len(self.emb.encode("ping"))
        ensure_collection(self.collection, dim)

        total = 0
        for batch in self.batched(docs, batch_size):
            texts = [d["text"] for d in batch]
            vecs = self.emb.encode(texts, batch_size=batch_size)
            points = [
                models.PointStruct(
                    id=str(uuid4()),
                    vector=vecs[i].tolist(),
                    payload={"page_content": texts[i], **(batch[i].get("metadata") or {})},
                )
                for i in range(len(batch))
            ]
            qclient.upsert(collection_name=self.collection, points=points, wait=wait)
            total += len(batch)
        return total

    def upsert(self, batch_size: int = 16, wait: bool = False) -> int:
        docs_list = df_to_docs(self.df, self.TEXT_COL, self.META_COLS)
        n = self.upsert_docs(docs_list, batch_size=batch_size, wait=wait)
        print("Upserted:", n)
        print("Count now:", qclient.count(self.collection, exact=True).count)
        return n
    
    def searching(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.emb.encode([query])[0].tolist()
        results = qclient.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k,
        )
        return results

# ---- usage (adjust path!) ----
if __name__ == "__main__":
    TEXT_COL = "TranslatedInstructions"
    META_COLS = ["TranslatedIngredients","Cuisine","TotalTimeInMins","URL","image-url","Ingredient-count"]

    df = pd.read_csv(r"D:\Chef_Intelligence\output_sample.csv")  # raw string for Windows path
    manager = EmbeddingManager(
        collection="food",
        emb="Snowflake/snowflake-arctic-embed-s",  # 384-dim matches your earlier setup
        df=df,
        TEXT_COL=TEXT_COL,
        META_COLS=META_COLS,
    )
    manager.upsert(batch_size=16, wait=False)
    query = "How to make pasta?"
    results = manager.searching(query, top_k=3)
    for res in results:
        print(f"Score: {res.score:.4f}, Content: {res.payload.get('page_content','')[:100]}...")
