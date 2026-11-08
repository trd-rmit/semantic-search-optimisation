"""Bedrock embedding and rerank calls. Extra keys in a model config are passed straight to the model API."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
from botocore.config import Config

_RETRY = Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=16)


def embed(texts, cfg, input_type, region):
    """input_type: 'search_document' or 'search_query' (used by Cohere and Nova, ignored by Titan)."""
    rt = boto3.client("bedrock-runtime", region_name=region, config=_RETRY)
    model, extra = cfg["model"], {k: v for k, v in cfg.items() if k not in ("model", "rpm")}
    call = lambda body: json.loads(rt.invoke_model(modelId=model, body=json.dumps(body))["body"].read())

    def one_per_request(body_of):
        if rpm := cfg.get("rpm"):  # account quota in requests/minute: pace instead of getting throttled
            out = []
            for t in texts:
                start = time.monotonic()
                out.append(call(body_of(t)))
                time.sleep(max(0.0, 60 / rpm - (time.monotonic() - start)))
            return out
        # shortcut: one request per text, 8 in parallel; use Bedrock batch inference for large corpora.
        with ThreadPoolExecutor(8) as pool:
            return list(pool.map(lambda t: call(body_of(t)), texts))

    if model.startswith("amazon.titan-embed"):
        vecs = [r["embedding"] for r in one_per_request(lambda t: {"inputText": t, **extra})]
    elif model.startswith("amazon.nova"):
        purpose = "GENERIC_INDEX" if input_type == "search_document" else "TEXT_RETRIEVAL"
        vecs = [r["embeddings"][0]["embedding"] for r in one_per_request(lambda t: {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {"embeddingPurpose": purpose, "text": {"truncationMode": "END", "value": t}, **extra}})]
    elif model.startswith("cohere.embed"):
        if "-v3" in model:  # Bedrock rejects Cohere v3 texts over 2,048 characters, so cut the end ourselves
            texts = [t[:2048] for t in texts]
        vecs = []
        for i in range(0, len(texts), 96):  # Cohere accepts at most 96 texts per call
            out = call({"texts": texts[i:i + 96], "input_type": input_type, **extra})["embeddings"]
            vecs += out["float"] if isinstance(out, dict) else out  # v4 / embedding_types returns a dict
    else:
        raise ValueError(f"unsupported embedding model: {model}")
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rerank(query, docs, cfg, region):
    """Returns indices of docs, best first."""
    ag = boto3.client("bedrock-agent-runtime", region_name=region, config=_RETRY)
    extra = {k: v for k, v in cfg.items() if k != "model"}
    model_cfg = {"modelArn": f"arn:aws:bedrock:{region}::foundation-model/{cfg['model']}"}
    if extra:
        model_cfg["additionalModelRequestFields"] = extra
    r = ag.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        sources=[{"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": d}}} for d in docs],
        rerankingConfiguration={"type": "BEDROCK_RERANKING_MODEL", "bedrockRerankingConfiguration": {
            "numberOfResults": len(docs), "modelConfiguration": model_cfg}},
    )
    return [x["index"] for x in r["results"]]
