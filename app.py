# app.py — Streamlit Cloud version (OpenAI SDK v2.x)
# --------------------------------------------------
# Generates a descriptive name and factual explanation for each keyword cluster.
# Works with openai>=2.0 (uses chat.completions.create + response_format).

import os
import re
import json
import time
import hashlib
import importlib.metadata
import pandas as pd
import streamlit as st
from openai import OpenAI

# --------------------------------------------------
# 🔹 Streamlit setup
# --------------------------------------------------
st.set_page_config(page_title="Cluster Describer (OpenAI)", layout="wide")
st.title("🔎 Cluster Describer — name + explanation generator")
st.caption("Upload a CSV with columns: cluster | keyword (comma-separated keywords per cluster) | search volume")

# --------------------------------------------------
# 🔹 API key (from Streamlit secrets)
# --------------------------------------------------
try:
    api_key = st.secrets["openai"]["api_key"]
except Exception:
    st.error("❌ Missing API key. Please add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets.\n\n```\n[openai]\napi_key = \"sk-...\"\n```")
    st.stop()

client = OpenAI(api_key=api_key)

# --------------------------------------------------
# 🔹 Version info
# --------------------------------------------------
try:
    ver = importlib.metadata.version("openai")
    st.sidebar.info(f"🧩 OpenAI SDK version: {ver}")
except Exception:
    st.sidebar.warning("Could not detect OpenAI version")

# --------------------------------------------------
# 🔹 Parameters
# --------------------------------------------------
model = "gpt-4o-mini-2024-07-18"
temperature = 0.1
sample_rows = st.slider("Preview / test on first N rows (0 = all)", 0, 200, 50, 10)

# --------------------------------------------------
# 🔹 Upload CSV
# --------------------------------------------------
file = st.file_uploader("Upload CSV", type=["csv"])
st.markdown("---")

# --------------------------------------------------
# 🔹 Helper functions
# --------------------------------------------------
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
            "description": "A short, human-readable topic name (≤5 words). Singular, neutral, no year.",
            "maxLength": 60,
        },
        "explanation": {
            "type": "string",
            "description": "One concise, factual sentence describing what the page covers (20–35 words). UK English.",
            "maxLength": 220,
        },
    },
    "required": ["descriptive_name", "explanation"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an SEO strategist. Given a page-level cluster and its keywords, "
    "produce a clear, factual 'descriptive_name' and one-sentence 'explanation' of the topic. "
    "Avoid sales or marketing tone. Use UK English."
)

def build_user_prompt(cluster: str, keywords: list[str]) -> str:
    return (
        f"Cluster (top keyword): {cluster.strip()}\n\n"
        f"Keywords: {json.dumps(keywords, ensure_ascii=False)}\n\n"
        "Return only a JSON object with 'descriptive_name' and 'explanation'."
    )

# --------------------------------------------------
# 🔹 Caching setup
# --------------------------------------------------
CACHE_DIR = ".cache_cluster_describer"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_key(cluster: str, keywords: list[str]) -> str:
    payload = json.dumps({"cluster": cluster, "keywords": keywords}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def cached_call(cluster: str, keywords: list[str]):
    """Call OpenAI API (cached)"""
    key = cache_key(cluster, keywords)
    path = os.path.join(CACHE_DIR, key + ".json")

    # Return cached if exists
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --- Call Chat Completions API (v2.x syntax) ---
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
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

        # Extract structured JSON result
        text = completion.choices[0].message.content
        data = json.loads(text)

    except Exception as e:
        st.warning(f"⚠️ API call failed for {cluster[:60]} — {e}")
        data = {
            "descriptive_name": cluster.strip().title(),
            "explanation": f"A page describing {cluster.strip()} and related keywords.",
        }

    # Cache result
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data

# --------------------------------------------------
# 🔹 Main app logic
# --------------------------------------------------
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

    st.success(f"Loaded {len(df)} clusters. Processing {len(df_run)} rows…")
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

    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results", csv, "cluster_descriptions.csv", "text/csv")
else:
    st.info("Upload a CSV to begin.")





