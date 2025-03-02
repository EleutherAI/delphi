import json
from pathlib import Path

import faiss
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from delphi.config import CacheConfig
from delphi.logger import logger


def get_neighbors(model, index, query: str, k: int = 1000):
    q_embedding = model.encode([query])
    result = index.search(q_embedding, k=k)
    # result: tuple of (L2 distances, top match indices).
    # supports matrix indexing for some reason so the top match index
    # requires two indices
    result[1][0][0]
    # text_data[first_result]

    # Remove the first result (which will be the query vector itself)
    # return distances[0][1:], neighbor_ids[0][1:]


def get_index_path(base_path: Path, cfg: CacheConfig):
    pretty_repo_name = cfg.dataset_repo.replace("/", "_")
    name = f"{pretty_repo_name}_{cfg.dataset_split}_{cfg.cache_ctx_len}.faiss"
    return base_path / name


def load_index(base_path: Path, cfg: CacheConfig) -> faiss.IndexFlatL2:
    index_path = get_index_path(base_path, cfg)
    return faiss.read_index(str(index_path))


def save_index(index: faiss.IndexFlatL2, base_path: Path, cfg: CacheConfig):
    index_path = get_index_path(base_path, cfg)

    faiss.write_index(index, str(index_path))

    with open(index_path.with_suffix(".json"), "w") as f:
        json.dump(
            {
                "index_path": str(index_path),
                "embedding_model": cfg.faiss_embedding_model,
            },
            f,
        )


def build_semantic_index(data: Dataset, cfg: CacheConfig, batch_size: int = 1024):
    """
    Build a semantic index, assuming data['text'] is of appropriate length.
    """

    model = SentenceTransformer(cfg.faiss_embedding_model, device="cuda")
    d = model[1].word_embedding_dimension

    index = faiss.IndexHNSWFlat(d, cfg.faiss_hnsw_config["M"])
    index.hnsw.efConstruction = cfg.faiss_hnsw_config["efConstruction"]
    index.hnsw.efSearch = cfg.faiss_hnsw_config["efSearch"]

    text_data = data["text"]

    embeddings = []
    for i in range(0, len(text_data), batch_size):
        print(f"Processing batch {i} of {len(text_data)}")
        batch = text_data[i : i + batch_size]
        batch_embeddings = model.encode(
            batch, batch_size=batch_size, device="cuda", convert_to_numpy=True
        )
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)

    index.add(embeddings)  # type: ignore

    return index


def build_or_load_index(data: Dataset, base_path: Path, cfg: CacheConfig):
    index_path = get_index_path(base_path, cfg)

    if not index_path.exists():
        logger.info(
            f"Building semantic index for {cfg.dataset_repo} {cfg.dataset_split}"
            "seq_len={cfg.example_ctx_len}..."
        )
        index = build_semantic_index(data, cfg)
        save_index(index, base_path, cfg)
        return index
    else:
        return load_index(base_path, cfg)


if __name__ == "__main__":
    from datasets import load_dataset

    from delphi.config import CacheConfig

    data = load_dataset("EleutherAI/fineweb-edu-dedup-10b", split="train[:1%]")
    cfg = CacheConfig(
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:1%]",
        ctx_len=256,
    )
    build_semantic_index(data, cfg)
