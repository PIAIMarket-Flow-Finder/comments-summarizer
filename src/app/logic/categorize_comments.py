import os
import joblib
import numpy as np
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Dict, Any

# Defines the output schema expected by the API
class CommentsOut(BaseModel):
    comments: List[Dict[str, Any]]  # ou adapte selon le contenu exact

# Loads the trained model once and caches it for reuse
@lru_cache
def _load_model():
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "comments_categorizer_model_xgb1.joblib")
    )
    return joblib.load(model_path)

# Return comments with prediction
def categorize_comments(*, comments: List[Dict[str, Any]]) -> CommentsOut:
    if not comments:
        return CommentsOut(comments=[])

    X = np.asarray([c["vector"] for c in comments], dtype=float)
    preds = _load_model().predict(X)

    for c, y in zip(comments, preds):
        c["category"] = int(y)     

    return CommentsOut(comments=comments)