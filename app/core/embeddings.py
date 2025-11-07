# embedding_manager.py
from typing import Iterable, List, Dict, Any, Union
from uuid import uuid4
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import models

from utils.custom_qdrant_client import qclient, ensure_collection   # function import OK now
from utils.metadata_handler import df_to_docs
from utils.text_processor import TextProcessor

class EmbeddingManager:
    def __init__(self, collection: str, emb: Union[str, SentenceTransformer], df: pd.DataFrame, TEXT_COL: str, META_COLS: List[str],):
        self.collection = collection
        self.emb = SentenceTransformer(emb) if isinstance(emb, str) else emb
        self.df = df.fillna("")
        self.TEXT_COL = TEXT_COL
        self.META_COLS = META_COLS

        # text processor uses defaults; you can tune chunk_size/overlap
        self.text_processor = TextProcessor(chunk_size=900, overlap=200)

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
        """
        docs: list of dicts: {"text": <str>, "metadata": {...}, optionally "row_id": <int>}
        This method will:
          - chunk each doc via semantic_chunking using self.emb
          - create chunk-level documents with metadata row_id & chunk_id
          - embed and upsert chunks in batches
        """
        docs = [d for d in docs if d.get("text")]
        if not docs:
            return 0

        # infer embedding dimension and ensure collection exists (non-destructive)
        dim = len(self.emb.encode("ping"))
        ensure_collection(self.collection, dim)

        # build chunked items: list of (row_id, chunk_id, chunk_text, metadata)
        chunked_items = []
        for row_idx, d in enumerate(docs):
            text = d["text"]
            meta = d.get("metadata", {}) or {}
            clean = self.text_processor.clean_text(text)

            # semantic chunking using encoder function wrapper
            def encoder_fn(sentences: List[str]):
                # return numpy array
                arr = self.emb.encode(sentences, batch_size=32, convert_to_numpy=True)
                return np.asarray(arr)

            try:
                chunks = self.text_processor.semantic_chunking(clean, encoder_fn=encoder_fn, similarity_threshold=0.65)
            except Exception:
                # fallback to simple char chunking if semantic fails for any reason
                chunks = self.text_processor.char_chunking(clean)

            # attach metadata and stable row id
            row_id = meta.get("row_id", meta.get("id", row_idx))
            for chunk_idx, chunk_text in enumerate(chunks):
                # you can add original text, but avoid duplicating huge fields
                chunk_meta = dict(meta)  # shallow copy
                chunk_meta.update({"row_id": row_id, "chunk_id": chunk_idx})
                chunked_items.append((row_id, chunk_idx, chunk_text, chunk_meta))

        # Now embed+upsert in batches
        total = 0
        for batch in self.batched(chunked_items, batch_size):
            texts = [item[2] for item in batch]
            # embed dense vectors (use convert_to_numpy to ensure numpy ndarray)
            vecs = self.emb.encode(texts, batch_size=batch_size, convert_to_numpy=True)

            points = []
            for i, (row_id, chunk_idx, txt, meta) in enumerate(batch):
                pts = models.PointStruct(
                    id=str(uuid4()),
                    vector=vecs[i].tolist(),
                    payload={"page_content": txt, **meta},
                )
                points.append(pts)

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
        query_vec = self.emb.encode([query], convert_to_numpy=True)[0].tolist()
        results = qclient.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
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
