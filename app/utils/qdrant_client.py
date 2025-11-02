# utils/qdrant_client.py
from qdrant_client import QdrantClient, models
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

qclient = QdrantClient(
    host="localhost",
    port=6333,
    #grpc_port=6334,
    prefer_grpc=False,
    timeout=300.0,
)

# def ensure_collection(collection_name: str, dim: int, distance: models.Distance = models.Distance.COSINE) -> None:
#     """
#     Ensure a simple dense collection exists with the given dimension and distance.
#     This does NOT create hybrid/sparse vectors — only a single unnamed dense vector.
#     If an existing collection has a different vector size or distance it will be deleted and recreated.
#     """
#     # If collection exists, validate config
#     if qclient.collection_exists(collection_name):
#         info = qclient.get_collection(collection_name)
#         vectors = info.config.params.vectors

#         # `vectors` can be a single VectorParams or a dict of named vectors.
#         if isinstance(vectors, dict):
#             # If someone accidentally created a named-vector (hybrid/named) collection,
#             # we treat it as incompatible and recreate.
#             need_recreate = True
#         else:
#             # vectors is a VectorParams
#             need_recreate = (vectors.size != dim) or (getattr(vectors, "distance", None) != distance)

#         if not need_recreate:
#             return  # collection is compatible — nothing to do

#         # incompatible config: delete and recreate
#         try:
#             qclient.delete_collection(collection_name)
#         except Exception as e:
#             # best-effort: warn but continue to attempt create (or re-raise if you prefer)
#             print(f"Warning deleting collection {collection_name}: {e}")

#     # Create simple dense collection (unnamed single vector)
#     qclient.create_collection(
#         collection_name=collection_name,
#         vectors_config=models.VectorParams(size=dim, distance=distance),
#     )


def ensure_collection(collection_name: str, dim: int, distance: models.Distance = models.Distance.COSINE) -> None:
    """
    Ensure a simple dense collection exists.
    - If it already exists, it will be reused as-is (no deletion or validation).
    - If not, it will be created with the given dimension and distance.
    """
    try:
        if qclient.collection_exists(collection_name):
            print(f"Collection '{collection_name}' already exists — using existing one.")
            return

        # Create if it doesn't exist
        qclient.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=dim, distance=distance),
        )
        print(f"Created new collection '{collection_name}' (dim={dim}, distance={distance}).")

    except Exception as e:
        print(f"Error ensuring collection '{collection_name}': {e}")
