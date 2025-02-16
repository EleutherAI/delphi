import json
from pathlib import Path

import faiss
from datasets import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from delphi.config import CacheConfig
from delphi.logger import logger


def get_neighbors_by_id(index: faiss.IndexIDMap, vector_id: int, k: int = 10):
    # First reconstruct the vector for the given ID
    vector = index.reconstruct(vector_id)

    # Reshape to match FAISS expectations (needs 2D array)
    vector = vector.reshape(1, -1)

    # Search for nearest neighbors
    distances, neighbor_ids = index.search(
        vector, k + 1
    )  # k+1 since it will find itself

    # Remove the first result (which will be the query vector itself)
    return distances[0][1:], neighbor_ids[0][1:]


def get_index_path(base_path: Path, cfg: CacheConfig):
    pretty_repo_name = cfg.dataset_repo.replace("/", "_")
    name = f"{pretty_repo_name}_{cfg.dataset_split}_{cfg.ctx_len}.faiss"
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
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            f,
        )


# def split_text(text: str, cfg: CacheConfig):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=cfg.ctx_len, chunk_overlap=cfg.ctx_len // 4
#     )
#     return splitter.split_text(text)


def split_text(text: str, cfg: CacheConfig):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.ctx_len,
        chunk_overlap=cfg.ctx_len // 4,
        length_function=lambda x: 1 + len(x) // 4,
    )
    return splitter.split_text(text)


def build_semantic_index(data: Dataset, cfg: CacheConfig):
    """
    Build a semantic index of the token sequences.
    """

    model = SentenceTransformer(cfg.faiss_embedding_model, device="cuda")
    d = next(model.parameters()).dtype

    index = faiss.IndexHNSWFlat(d, cfg.faiss_hnsw_config["M"])
    index.hnsw.efConstruction = cfg.faiss_hnsw_config["efConstruction"]
    index.hnsw.efSearch = cfg.faiss_hnsw_config["efSearch"]

    data["text"]
    breakpoint()

    # index_tokenizer = AutoTokenizer.from_pretrained
    # ('sentence-transformers/all-MiniLM-L6-v2')
    # index_model = AutoModel.from_pretrained(
    # 'sentence-transformers/all-MiniLM-L6-v2').to("cuda")

    # index_tokens = chunk_and_tokenize(data, index_tokenizer, max_seq_len=cfg.ctx_len,
    # text_key=cfg.dataset_row)
    # index_tokens = index_tokens["input_ids"]
    # index_tokens = assert_type(Tensor, index_tokens)

    # token_embeddings = index_model(index_tokens[:2].to("cuda")).last_hidden_state

    # base_index = faiss.IndexFlatL2(token_embeddings.shape[-1])
    # index = faiss.IndexIDMap(base_index)

    # batch_size = 512
    # dataloader = DataLoader(index_tokens, batch_size=batch_size) # type: ignore

    # from tqdm import tqdm
    # with torch.no_grad():
    #     for batch_idx, batch in enumerate(tqdm(dataloader)):
    #         batch = batch.to("cuda")
    #         token_embeddings = index_model(batch).last_hidden_state
    #         sentence_embeddings = token_embeddings.mean(dim=1)
    #         sentence_embeddings = sentence_embeddings.cpu().numpy().astype(np.float32)

    #         ids = np.arange(batch_idx * batch_size, batch_idx * batch_size +
    # len(batch))
    #         index.add_with_ids(sentence_embeddings, ids)

    return None
    # """
    # Build a semantic index of the token sequences.
    # """

    # model = SentenceTransformer(cfg.faiss_embedding_model, device="cuda")
    # d = next(model.parameters()).dtype

    # text = data['text']
    # chunks = []
    # for t in text:
    #     chunks.extend(split_text(t, cfg))

    # breakpoint()
    # index = faiss.IndexHNSWFlat(d, cfg.faiss_hnsw_config["M"])
    # index.metric_type = faiss.METRIC_L2
    # index.hnsw.efConstruction = cfg.faiss_hnsw_config["efConstruction"]
    # index.hnsw.efSearch = cfg.faiss_hnsw_config["efSearch"]

    # index_tokenizer = AutoTokenizer.from_pretraine
    # d('sentence-transformers/all-MiniLM-L6-v2')
    # index_model = AutoModel.from_pretrained('sentence-transform
    # ers/all-MiniLM-L6-v2').to("cuda")

    # index_tokens = chunk_and_tokenize(data, index_tokenizer,
    # max_seq_len=cfg.ctx_len, text_key=cfg.dataset_row)
    # index_tokens = index_tokens["input_ids"]
    # index_tokens = assert_type(Tensor, index_tokens)

    # token_embeddings = index_model(index_tokens[:2].to("cuda")).last_hidden_state

    # base_index = faiss.IndexFlatL2(token_embeddings.shape[-1])
    # index = faiss.IndexIDMap(base_index)

    # batch_size = 512
    # dataloader = DataLoader(index_tokens, batch_size=batch_size) # type: ignore

    # from tqdm import tqdm
    # with torch.no_grad():
    #     for batch_idx, batch in enumerate(tqdm(dataloader)):
    #         batch = batch.to("cuda")
    #         token_embeddings = index_model(batch).last_hidden_state
    #         sentence_embeddings = token_embeddings.mean(dim=1)
    #         sentence_embeddings = sentence_embeddings.cpu().numpy()
    # .astype(np.float32)

    #         ids = np.arange(batch_idx * batch_size, batch_idx * batch_size
    # + len(batch))
    #         index.add_with_ids(sentence_embeddings, ids)

    # return None


def build_or_load_index(data: Dataset, base_path: Path, cfg: CacheConfig):
    index_path = get_index_path(base_path, cfg)

    if not index_path.exists():
        logger.info(
            f"Building semantic index for {cfg.dataset_repo} {cfg.dataset_split}"
            "seq_len={cfg.ctx_len}..."
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
