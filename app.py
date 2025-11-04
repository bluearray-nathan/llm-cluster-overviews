# app.py — Streamlit Cloud Optimized
# ------------------------------------------------
# Generates a descriptive name and factual explanation for each keyword cluster
# Uses OpenAI Responses API (requires openai>=1.3.0, ideally >=1.51.0)

import os
import re
import json
import time
import hashlib
import importlib
import pandas as pd
import streamlit as st
from openai import OpenAI

# ------------------------------------------------
# 🔹 Page setup
# ------------------------------------------------
st.set_page_config(page_title="Cluster Describer (OpenAI)", layout="wide")
st.title("🔎 Cluster Describer — Name & Explanation Generator")
st.caption("Upload a CSV with columns: `cluster`, `keyword`, `search volume`")

# ------------------------------------------------
# 🔹 API key from secrets.toml or Streamlit Cloud Secrets
# ------------------------------------------------
try:
    api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing API key. Please add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets:\n\n```\n[openai]\napi_key = \"sk-...\"\n```")
    st.stop()

client = OpenAI(api_key=api_key)

# ------------------------------------------------
# 🔹 Version check & fallback
# ------------------------------------------------
try:
    openai_pkg = importlib.metadata.version("openai")
    openai_major = int(openai_pkg.split(".")[0])
    if openai_major < 1:
        st.warning(f"Your openai package version ({openai_pkg}) is outdated. Update to >=1.3.0 for Responses API support.")
except Exception:
    st.info("Could not check OpenAI version; continuing...")

# ------------------------------------------------
# 🔹 Parameters
# ------------------------------------------------
model = "gpt-4o-mini-2024-07-18"
temperature = 0.1
sample_rows = st.slider("Preview / test on first N rows (0 = all)", 0, 200, 50, 10)

# ------------------------------------------------
# 🔹 Upload CSV
# ------------------------------------------------
file = st.file_uploader("Upload CSV", type=["csv"])
st.markdown("---")

# ------------------------------------------------
# 🔹 Helper functions
# ------------------------------------------------
def clean_keywords_str(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def split_keywords(s: str, top_k=30):
    raw = [k.strip() for k in str(s).split(",") if k.strip()]
    seen, out = set(), []
    for k in raw:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            out.append(k)
        if len(out) >= top_k:
            break
    return out

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "descriptive_name": {
            "type": "string",
            "description": "Short, human-readable topic name (≤5 words). Singular, neutral, no year.",
            "maxLength": 60
        },
        "explanation": {
            "type": "string",
            "description": "One concise, factual sentence describing what the page covers (20–35 words). UK English.",
            "maxLength": 220
        }
    },
    "required": ["descriptive_name", "explanation"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an SEO strategist. Given a page-level cluster and its keywords, "
    "produce a clear, factual descriptive_name and one-sentence explanation of the topic. "
    "Avoid sales language. Use UK English."
)

def build_user_prompt(cluster: str, keywords: list[str]) -> str:
    return (
        f"Cluster (top keyword): {cluster.strip()}\n\n"
        f"Keywords: {json.dumps(keywords, ensure_ascii=False)}\n\n"
        "Return a descriptive_name and one-sentence factual explanation."
    )

# ------------------------------------------------
# 🔹 Cache setup
# ------------------------------------------------
CACHE_DIR = ".cache_cluster_describer"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_key(cluster: str, keywords: list[str]) -> str:
    payload = json.dumps({"cluster": cluster, "keywords": keywords}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def cached_call(cluster: str, keywords: list[str]):
    key = cache_key(cluster, keywords)
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --- OpenAI Responses API call ---
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(cluster, keywords)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ClusterLabel",
                    "schema": JSON_SCHEMA,
                    "strict": True,
                },
            },
            temperature=temperature,
        )

        # Extract text output
        data = {}
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    data = json.loads(c.text)
                    break

    except TypeError as e:
        st.error("⚠️ Your environment’s OpenAI package doesn’t support `response_format`. "
                 "Please upgrade to `openai>=1.3.0` in requirements.txt and redeploy.")
        st.stop()
    except Exception as e:
        st.warning(f"OpenAI API call failed for {cluster[:50]}: {e}")
        data = {}

    # Fallback
    if not isinstance(data, dict) or "descriptive_name" not in data:
        data = {
            "descriptive_name": cluster.strip().title(),
            "explanation": f"A page describing {cluster.strip()} and related search keywords."
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data

# ------------------------------------------------
# 🔹 Main App
# ------------------------------------------------
if file:
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    cluster_col = cols.get("cluster")
    keyword_col = cols.get("keyword")

    if not cluster_col or not keyword_col:
        st.error("CSV must include 'cluster' and 'keyword' columns.")
        st.stop()

    df["_keywords_list"] = df[keyword_col].map(clean_keywords_str).map(split_keywords)
    df = df.drop_duplicates(subset=[cluster_col]).reset_index(drop=True)
    df_run = df.head(sample_rows) if sample_rows > 0 else df

    st.success(f"Loaded {len(df)} clusters. Processing {len(df_run)} rows.")
    progress = st.progress(0)
    results = []

    for i, row in df_run.iterrows():
        cluster_name = str(row[cluster_col])
        kw_list = list(row["_keywords_list"])
        result = cached_call(cluster_name, kw_list)
        results.append({
            "cluster": cluster_name,
            "descriptive_name": result.get("descriptive_name", ""),
            "explanation": result.get("explanation", "")
        })
        progress.progress((i + 1) / len(df_run))
        time.sleep(0.05)

    progress.empty()
    out_df = pd.DataFrame(results)
    st.subheader("Preview")
    st.dataframe(out_df.head(50), use_container_width=True)

    st.download_button(
        "📥 Download Results",
        out_df.to_csv(index=False).encode("utf-8"),
        "cluster_descriptions.csv",
        "text/csv"
    )
else:
    st.info("Upload a CSV to begin.")




