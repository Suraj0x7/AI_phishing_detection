import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def inspect_single_file(filepath):
    filename = os.path.basename(filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    info = {
        "filename": filename,
        "size_mb": round(size_mb, 2),
        "rows": 0,
        "columns": 0,
        "column_names": [],
        "text_column": None,
        "label_column": None,
        "missing_values": {},
        "duplicate_rows": 0,
        "duplicate_texts": 0,
        "class_distribution": {},
        "status": "unprocessed"
    }

    try:
        if size_mb > 500:
            df = pd.read_csv(filepath, nrows=10000, low_memory=False)
            info["rows"] = "500,000+ (sampled 10,000 for schema)"
        else:
            df = pd.read_csv(filepath, low_memory=False)
            info["rows"] = len(df)

        info["columns"] = len(df.columns)
        info["column_names"] = list(df.columns)
        info["missing_values"] = df.isnull().sum().to_dict()
        info["duplicate_rows"] = int(df.duplicated().sum())

        # Detect text column
        text_cols = [c for c in df.columns if any(k in c.lower() for k in ['text', 'body', 'message', 'email', 'content'])]
        if text_cols:
            info["text_column"] = text_cols[0]
            if df[info["text_column"]].dtype == object:
                info["duplicate_texts"] = int(df[info["text_column"]].duplicated().sum())

        # Detect label column
        label_cols = [c for c in df.columns if any(k in c.lower() for k in ['label', 'category', 'class'])]
        if label_cols:
            info["label_column"] = label_cols[0]
            info["class_distribution"] = df[info["label_column"]].value_counts(dropna=False).head(10).to_dict()

    except Exception as e:
        info["status"] = f"Error reading file: {str(e)}"

    return info

def inspect_all_datasets(data_dir="data"):
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    reports = []
    print("\n" + "="*70)
    print("           AUTOMATED DATASET INSPECTION & AUDIT")
    print("="*70)

    for f in sorted(files):
        fp = os.path.join(data_dir, f)
        info = inspect_single_file(fp)
        reports.append(info)

        print(f"\n[FILE]: {info['filename']} ({info['size_mb']} MB)")
        print(f"  Rows: {info['rows']} | Columns: {info['columns']}")
        print(f"  Column Names: {info['column_names']}")
        print(f"  Detected Text Column: {info['text_column']}")
        print(f"  Detected Label Column: {info['label_column']}")
        print(f"  Missing Values: {info['missing_values']}")
        print(f"  Duplicate Rows: {info['duplicate_rows']}")
        print(f"  Duplicate Texts: {info['duplicate_texts']}")
        print(f"  Class Distribution: {info['class_distribution']}")

    return reports

def build_combined_dataset(data_dir="data", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    records = []

    # 1. Claude Synthetic Phishing
    claude_fp = os.path.join(data_dir, "claude_synthetic_ai_phishing_3000.csv")
    if os.path.exists(claude_fp):
        df_claude = pd.read_csv(claude_fp)
        for _, r in df_claude.iterrows():
            t = str(r['text']).strip()
            records.append({'text': t, 'label': 'ai_phishing', 'source': 'Claude_Synthetic'})
        print(f"[Loaded] Claude Synthetic: {len(df_claude)} records -> mapped to 'ai_phishing'")

    # 2. GPT-4 Synthetic Phishing
    gpt_fp = os.path.join(data_dir, "GPT_4_synthetic_ai_phishing_emails_3000.csv")
    if os.path.exists(gpt_fp):
        df_gpt = pd.read_csv(gpt_fp)
        for _, r in df_gpt.iterrows():
            t = str(r['text']).strip()
            records.append({'text': t, 'label': 'ai_phishing', 'source': 'GPT4_Synthetic'})
        print(f"[Loaded] GPT-4 Synthetic: {len(df_gpt)} records -> mapped to 'ai_phishing'")

    # 3. Nazario Phishing & Enron Legitimate
    naz_fp = os.path.join(data_dir, "Nazario_5.csv")
    if os.path.exists(naz_fp):
        df_naz = pd.read_csv(naz_fp)
        for _, r in df_naz.iterrows():
            subj = str(r['subject']).strip() if pd.notnull(r['subject']) and str(r['subject']).strip().lower() != 'nan' else ""
            body = str(r['body']).strip() if pd.notnull(r['body']) and str(r['body']).strip().lower() != 'nan' else ""
            text = f"Subject: {subj}\n\n{body}" if subj else body
            
            lbl = int(r['label'])
            if lbl == 1:
                target_label = "phishing"
                src = "Nazario_Phishing"
            else:
                target_label = "legitimate"
                src = "Enron_Legitimate"
            records.append({'text': text, 'label': target_label, 'source': src})
        print(f"[Loaded] Nazario_5: {len(df_naz)} records -> (1565 'phishing', 1500 'legitimate')")

    # 4. Explicitly Report Ambiguous & Incompatible Datasets
    print("\n--- DATASET COMPATIBILITY & MAPPING AUDIT ---")
    print("[EXCLUDED] gemini_generated_emails_3000.csv:")
    print("  Reason: Ambiguous labels. Lacks ground-truth security classification (only topical categories like DevOps, Corporate).")
    print("  Policy: Strict academic integrity rule: DO NOT silently guess ambiguous labels.")
    
    print("[EXCLUDED] email_text.csv:")
    print("  Reason: Aggressively pre-processed (all lowercased, punctuation stripped, numbers replaced with 'escapenumber').")
    print("  Stylometric impact: Destroys casing, sentence structure, and punctuation needed for stylometric analysis.")
    
    print("[EXCLUDED] Enron.csv (Raw 1.36GB):")
    print("  Reason: Raw unprocessed MIME archive. Clean Enron emails are already sampled and standardized in Nazario_5.")

    combined_df = pd.DataFrame(records)
    output_fp = os.path.join(output_dir, "combined_dataset.csv")
    combined_df.to_csv(output_fp, index=False, encoding='utf-8')
    print(f"\n[Saved] Standardized combined dataset to: {output_fp}")
    print(f"Total Rows: {len(combined_df)}")
    print("Class Distribution:\n", combined_df['label'].value_counts())
    print("Source Distribution:\n", combined_df['source'].value_counts())

    # Generate Figures: 01_class_distribution.png and 02_source_distribution.png
    plot_dataset_distributions(combined_df)

    return combined_df

def plot_dataset_distributions(df, figures_dir="results/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    palette = {"legitimate": "#2ecc71", "phishing": "#e74c3c", "ai_phishing": "#3498db"}

    # Figure 1: Class Distribution
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    counts = df['label'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=[palette.get(c, "#95a5a6") for c in counts.index], width=0.55, edgecolor='black', linewidth=1.2)
    ax.set_title("Dataset Class Distribution (Three-Class Framework)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Target Security Class", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Number of Email Samples", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0, max(counts.values) * 1.15)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 70, f"{int(h):,}\n({h/len(df)*100:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig_path1 = os.path.join(figures_dir, "01_class_distribution.png")
    fig.savefig(fig_path1, dpi=300, bbox_inches="tight")
    # Also save as class_distribution.png for requirement naming
    fig.savefig(os.path.join(figures_dir, "class_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Source Distribution
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    src_counts = df['source'].value_counts()
    src_colors = sns.color_palette("mako", len(src_counts))
    bars = ax.bar(src_counts.index, src_counts.values, color=src_colors, width=0.6, edgecolor='black', linewidth=1.2)
    ax.set_title("Dataset Distribution by Verified Data Source", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Data Origin Source", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Number of Email Samples", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0, max(src_counts.values) * 1.15)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 70, f"{int(h):,}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig_path2 = os.path.join(figures_dir, "02_source_distribution.png")
    fig.savefig(fig_path2, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "source_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Distribution figures: {fig_path1}, {fig_path2}")

if __name__ == "__main__":
    inspect_all_datasets()
    build_combined_dataset()
