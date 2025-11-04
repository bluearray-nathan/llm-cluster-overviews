# app.py
# -----------------------------------
# Streamlit app to generate descriptive names & factual explanations for clusters
# Reads OpenAI API key from .streamlit/secrets.toml

import os
import json
import time
import re
import hashlib
import pandas as pd
import streamlit as st
from openai import OpenAI

# --- Streamlit setup ---
st.set_page_config(page_title="Cluster Describer (OpenAI)", layout="wide")
st.title("🔎 Cluster Describer — name + explanation (OpenAI)")
st.caption("Upload a CSV with columns: cluster | keyword (comma-separated keywords per cluster) | search volume")

# --- API setup (uses secrets.toml) ---
try:
    api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Please set your OpenAI API key in .streamlit/secrets.toml under [openai].")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Parameters ---
model = "gpt-4o-mini-2024-07-18"
temperature = 0.1

# --- Upload CSV ---
file = st.file_uploader("Upload CSV", type=["csv"])
sample_rows = st.slider("Preview / test on first N rows (0 = all)", 0, 200, 50, 10)
st.markdown("---")

# --- Helpers ---
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

# JSON Schema for structured response
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "descriptive_name": {
            "type": "string",
            "description": "A short, human-readable topic name (max 5 words). Singular, neutral, no year.",
            "maxLength": 60
        },
        "explanation": {
            "type": "string",
            "description": "One concise, factual sentence describing what the page is about based on the keywords (20–35 words). UK English.",
            "maxLength": 220
        }
    },
    "required": ["descriptive_name", "explanation"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an SEO strategist. Given a cluster name and its keywords, "
    "produce a clear, neutral label and a one-sentence factual explanation of the page topic. "
    "Avoid sales language or fluff. Use UK English."
)

def build_user_prompt(cluster: str, keywords: list[str]) -> str:
    return (
        f"Cluster (top keyword): {cluster.strip()}\n\n"
        f"Keywords: {json.dumps(keywords, ensure_ascii=False)}\n\n"
        "Return a descriptive_name and one-sentence explanation."
    )

# Cache
CACHE_DIR = ".cache_cluster_describer"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_key(cluster: str, keywords: list[str]) -> str:
    payload = json.dumps({"c": cluster, "k": keywords}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def cached_call(cluster: str, keywords: list[str]):
    key = cache_key(cluster, keywords)
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(cluster, keywords)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ClusterLabel", "schema": JSON_SCHEMA, "strict": True}
        },
        temperature=temperature,
    )

    # Extract JSON output
    data = {}
    try:
        for item in resp.output or []:
            for c in getattr(item, "content", []):
                if getattr(c, "type", None) in ("output_text", "text"):
                    data = json.loads(c.text)
                    break
    except Exception:
        pass

    if not isinstance(data, dict) or "descriptive_name" not in data:
        data = {
            "descriptive_name": cluster.strip().title(),
            "explanation": f"A page describing {cluster.strip()} based on related search keywords."
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data

# --- Main logic ---
if file:
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    cluster_col = cols.get("cluster")
    keyword_col = cols.get("keyword")

    if not cluster_col or not keyword_col:
        st.error("CSV must have 'cluster' and 'keyword' columns.")
        st.stop()

    df["_keywords_list"] = df[keyword_col].map(clean_keywords_str).map(split_keywords)
    df = df.drop_duplicates(subset=[cluster_col]).reset_index(drop=True)
    df_run = df.head(sample_rows) if sample_rows > 0 else df

    st.success(f"Loaded {len(df)} clusters. Processing {len(df_run)} rows...")
    progress = st.progress(0)
    out_rows = []

    for i, row in df_run.iterrows():
        cluster_name = str(row[cluster_col])
        kw_list = list(row["_keywords_list"])
        result = cached_call(cluster_name, kw_list)
        out_rows.append({
            "cluster": cluster_name,
            "descriptive_name": result.get("descriptive_name", ""),
            "explanation": result.get("explanation", "")
        })
        progress.progress((i + 1) / len(df_run))
        time.sleep(0.02)

    progress.empty()
    out_df = pd.DataFrame(out_rows)
    st.subheader("Preview")
    st.dataframe(out_df.head(50), use_container_width=True)

    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results", csv, "cluster_descriptions.csv", "text/csv")

else:
    st.info("Upload a CSV to begin.")



