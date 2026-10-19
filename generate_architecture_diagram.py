import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_system_architecture_diagram(out_png="results/figures/system_architecture.png", 
                                        out_svg="results/figures/system_architecture.svg",
                                        out_mmd="results/figures/system_architecture.mmd"):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # 1. Save Mermaid format
    mermaid_code = """
graph TD
    A[Data Sources: Enron, Nazario, Claude, GPT-4] --> B[Text Preprocessing & Sanitization]
    B --> C1[Stylometric Branch: 10 Primary Style Features]
    B --> C2[Semantic Branch: Sublinear Word N-Grams TF-IDF]
    C1 --> D1[StandardScaler: Training-Fitted Normalization]
    C2 --> D2[Sparse Matrix: 15,000 Dimensions]
    D1 --> E[Feature Fusion Layer: SciPy Sparse Concatenation]
    D2 --> E
    E --> F[Hybrid Feature Vector: 15,010 Dimensions]
    F --> G1[Logistic Regression]
    F --> G2[Random Forest]
    F --> G3[XGBoost]
    G1 --> H[Output Classification]
    G2 --> H
    G3 --> H
    H --> I1[1. Legitimate Email]
    H --> I2[2. Traditional Phishing]
    H --> I3[3. AI-Generated Phishing]
    H --> J[Stylometric Explainability: Z-Score Profiling & Analyst Signals]
    C1 -.-> J
"""
    with open(out_mmd, "w") as f:
        f.write(mermaid_code.strip())
    print(f"[Saved] Mermaid architecture to: {out_mmd}")

    # 2. Render high-res diagram via matplotlib
    fig, ax = plt.subplots(figsize=(14, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color scheme
    c_source = "#34495e"
    c_prep = "#2980b9"
    c_sty = "#8e44ad"
    c_sem = "#16a085"
    c_fusion = "#d35400"
    c_model = "#2c3e50"
    c_out = "#27ae60"
    c_exp = "#c0392b"

    def draw_box(x, y, w, h, title, subtitle="", color="#34495e", text_color="white"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.5",
            linewidth=1.5, edgecolor="black", facecolor=color, alpha=0.92
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)
            ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', fontsize=9, color=text_color, style='italic')
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            label, xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor='#2c3e50', edgecolor='#2c3e50', width=1.5, headwidth=7, shrink=0.08),
            ha='center', va='center', fontsize=8, fontweight='bold', color='#2c3e50'
        )

    # Title
    ax.text(50, 97.5, "HYBRID STYLOMETRIC & SEMANTIC PHISHING DETECTION FRAMEWORK",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#2c3e50')
    ax.text(50, 95.0, "End-to-End System Architecture for Detecting AI-Generated & Traditional Phishing",
            ha='center', va='center', fontsize=11, color='#7f8c8d')

    # Layer 1: Data Sources
    draw_box(15, 87, 70, 5.5, "Data Ingestion & Verification Layer", 
             "Enron (Legitimate) | Nazario (Traditional Phishing) | Claude & GPT-4 (Synthetic AI Phishing)", c_source)

    draw_arrow(50, 87, 50, 80)

    # Layer 2: Preprocessing
    draw_box(20, 74.5, 60, 5.5, "Safe Text Preprocessing (Style-Preserving)", 
             "URL & Email Tokenization | HTML Removal | Case & Punctuation Preservation", c_prep)

    # Split into two branches
    draw_arrow(35, 74.5, 25, 66)
    draw_arrow(65, 74.5, 75, 66)

    # Layer 3: Two Branches
    # Stylometric Branch
    draw_box(6, 59, 38, 7, "Stylometric Branch (10 Features)", 
             "Word, Char, Sentence, WordLen, Punct, Caps,\nPronoun Density, Urgency, Imperatives, Flesch FRE", c_sty)

    # Semantic Branch
    draw_box(56, 59, 38, 7, "Semantic Branch (TF-IDF)", 
             "Word Unigrams + Bigrams (1,2) | Sublinear TF\n15,000 Features (Fitted Strictly on Train Split)", c_sem)

    # Normalization / Scaling Layer
    draw_arrow(25, 59, 25, 52)
    draw_arrow(75, 59, 75, 52)

    draw_box(8, 46, 34, 5.5, "StandardScaler Normalization", "Fitted strictly on Training Stylometrics", c_sty)
    draw_box(58, 46, 34, 5.5, "Sparse Matrix Representation", "CSR Sparse Vector (15,000 Dimensions)", c_sem)

    # Convergence into Feature Fusion
    draw_arrow(25, 46, 45, 39)
    draw_arrow(75, 46, 55, 39)

    # Layer 4: Feature Fusion
    draw_box(25, 32.5, 50, 6, "Hybrid Feature Fusion Layer", 
             "Fused Sparse Representation: 15,010 Dimensions [Stylometric + TF-IDF]", c_fusion)

    # Layer 5: ML Classifiers
    draw_arrow(50, 32.5, 50, 26)
    draw_box(10, 19, 80, 7, "Multi-Class Machine Learning Classifiers", 
             "Logistic Regression (C=10.0, Balanced)  |  Random Forest (n=200)  |  XGBoost (n=200, depth=5)\nHyperparameters Optimized via 5-Fold Stratified Cross Validation on Train Split Only", c_model)

    # Output Classes
    draw_arrow(25, 19, 20, 12)
    draw_arrow(50, 19, 50, 12)
    draw_arrow(75, 19, 80, 12)

    draw_box(6, 6, 26, 5.5, "1. Legitimate Email", "Enron Workplace Archive", c_out)
    draw_box(37, 6, 26, 5.5, "2. Traditional Phishing", "Nazario Benchmark Corpus", "#e74c3c")
    draw_box(68, 6, 26, 5.5, "3. AI-Generated Phishing", "Claude & GPT-4 Synthetics", "#3498db")

    # Explainability Side Box
    exp_rect = patches.FancyBboxPatch(
        (82, 33), 15, 14, boxstyle="round,pad=0.5",
        linewidth=1.2, edgecolor="#c0392b", facecolor="#fadbd8", alpha=0.95
    )
    ax.add_patch(exp_rect)
    ax.text(89.5, 44.5, "Analyst Explainability", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#922b21')
    ax.text(89.5, 39.5, "Z-Score Profiling vs.\nTraining Class Norms\nRanked Stylometric\nRisk Signals", 
            ha='center', va='center', fontsize=7.5, color='#78281f')

    # Connecting dotted arrow from stylometry to explainability
    ax.annotate(
        "", xy=(89.5, 47), xytext=(44, 62.5),
        arrowprops=dict(arrowstyle="->", linestyle="dashed", color="#c0392b", lw=1.5)
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] System architecture figure: {out_png}")
    print(f"[Saved] System architecture SVG: {out_svg}")

if __name__ == "__main__":
    generate_system_architecture_diagram()
