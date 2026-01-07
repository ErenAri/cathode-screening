#!/usr/bin/env python
"""
Generate screening reports in HTML and PDF format.

Creates a professional report with ranked candidate table,
summary statistics, and actionable recommendations.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cathode Screening Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            color: #2d3748;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        .header h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
        .header p {{ opacity: 0.9; font-size: 0.95rem; }}
        
        /* Summary Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .card-label {{ font-size: 0.85rem; color: #718096; margin-bottom: 0.25rem; }}
        .card-value {{ font-size: 1.8rem; font-weight: 600; color: #2d3748; }}
        .card-value.green {{ color: #38a169; }}
        .card-value.blue {{ color: #3182ce; }}
        .card-value.orange {{ color: #dd6b20; }}
        
        /* Table */
        .table-container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
            margin-bottom: 2rem;
        }}
        .table-header {{
            background: #edf2f7;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        .table-header h2 {{ font-size: 1.1rem; color: #2d3748; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ 
            background: #f7fafc;
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.85rem;
            color: #4a5568;
            border-bottom: 2px solid #e2e8f0;
        }}
        td {{ 
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.9rem;
        }}
        tr:hover {{ background: #f7fafc; }}
        
        /* Action Badges */
        .action {{ 
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .action-dft {{ background: #c6f6d5; color: #22543d; }}
        .action-hold {{ background: #feebc8; color: #744210; }}
        .action-skip {{ background: #fed7d7; color: #822727; }}
        
        /* Uncertainty */
        .unc-low {{ color: #38a169; }}
        .unc-med {{ color: #dd6b20; }}
        .unc-high {{ color: #e53e3e; }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 1rem;
            color: #718096;
            font-size: 0.85rem;
        }}
        
        @media print {{
            body {{ background: white; }}
            .container {{ max-width: 100%; padding: 0; }}
            .header {{ background: #1a365d !important; -webkit-print-color-adjust: exact; }}
            .card {{ box-shadow: none; border: 1px solid #e2e8f0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔋 Cathode Screening Report</h1>
            <p>Generated: {timestamp} | Model: CGCNN Ensemble (K=5) | Dataset: {dataset_name}</p>
        </div>
        
        <div class="summary-grid">
            <div class="card">
                <div class="card-label">Total Candidates</div>
                <div class="card-value">{n_total}</div>
            </div>
            <div class="card">
                <div class="card-label">Recommended for DFT</div>
                <div class="card-value green">{n_dft}</div>
            </div>
            <div class="card">
                <div class="card-label">Hold for Review</div>
                <div class="card-value orange">{n_hold}</div>
            </div>
            <div class="card">
                <div class="card-label">Predicted Stable</div>
                <div class="card-value blue">{n_stable_pred}</div>
            </div>
        </div>
        
        <div class="table-container">
            <div class="table-header">
                <h2>📊 Ranked Candidates</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Material ID</th>
                        <th>Pred E<sub>hull</sub> (eV)</th>
                        <th>P(Stable)</th>
                        <th>Uncertainty</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Cathode Screening System v1.0 | Powered by CGCNN + Active Learning</p>
        </div>
    </div>
</body>
</html>
"""


def classify_uncertainty(std: float) -> str:
    """Classify uncertainty level."""
    if std < 0.05:
        return "Low"
    elif std < 0.15:
        return "Medium"
    else:
        return "High"


def get_action(p_stable: float, uncertainty: str, pred_ehull: float) -> str:
    """Determine recommended action."""
    if p_stable > 0.7 and uncertainty == "Low" and pred_ehull < 0.1:
        return "DFT"
    elif p_stable > 0.5 or pred_ehull < 0.15:
        return "HOLD"
    else:
        return "SKIP"


def generate_table_rows(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Generate HTML table rows."""
    rows = []
    
    for i, row in df.head(max_rows).iterrows():
        rank = row.get("rank", i + 1)
        mid = row.get("material_id", f"MP-{i}")
        pred_ehull = row.get("q50", row.get("pred_ehull", 0))
        p_stable = row.get("p_stable", row.get("pi", 0))
        std = row.get("epistemic_std", row.get("uncertainty", 0.1))
        
        unc_label = classify_uncertainty(std)
        action = get_action(p_stable, unc_label, pred_ehull)
        
        unc_class = {"Low": "unc-low", "Medium": "unc-med", "High": "unc-high"}[unc_label]
        action_class = {"DFT": "action-dft", "HOLD": "action-hold", "SKIP": "action-skip"}[action]
        
        rows.append(f"""
            <tr>
                <td>{rank}</td>
                <td><strong>{mid}</strong></td>
                <td>{pred_ehull:.3f}</td>
                <td>{p_stable:.2f}</td>
                <td class="{unc_class}">{unc_label}</td>
                <td><span class="action {action_class}">{action}</span></td>
            </tr>
        """)
    
    if len(df) > max_rows:
        rows.append(f"""
            <tr>
                <td colspan="6" style="text-align: center; color: #718096; font-style: italic;">
                    ... and {len(df) - max_rows} more candidates
                </td>
            </tr>
        """)
    
    return "\n".join(rows)


def generate_report(
    predictions_path: str,
    output_dir: str,
    dataset_name: str = "Cathode Materials",
    max_rows: int = 100,
) -> Path:
    """Generate HTML screening report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    df = pd.read_parquet(predictions_path)
    
    # Sort by predicted stability (lower E_hull = better)
    sort_col = "q50" if "q50" in df.columns else "pred_ehull"
    df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    # Calculate summary stats
    p_stable_col = "p_stable" if "p_stable" in df.columns else None
    std_col = "epistemic_std" if "epistemic_std" in df.columns else None
    
    n_dft = 0
    n_hold = 0
    n_skip = 0
    
    for _, row in df.iterrows():
        p_stable = row.get("p_stable", 0.5)
        std = row.get("epistemic_std", 0.1)
        pred = row.get("q50", row.get("pred_ehull", 0.5))
        
        unc = classify_uncertainty(std)
        action = get_action(p_stable, unc, pred)
        
        if action == "DFT":
            n_dft += 1
        elif action == "HOLD":
            n_hold += 1
        else:
            n_skip += 1
    
    n_stable_pred = (df[sort_col] < 0.05).sum() if sort_col in df.columns else 0
    
    # Generate HTML
    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        dataset_name=dataset_name,
        n_total=len(df),
        n_dft=n_dft,
        n_hold=n_hold,
        n_stable_pred=n_stable_pred,
        table_rows=generate_table_rows(df, max_rows),
    )
    
    # Save HTML
    html_path = output_dir / "screening_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"HTML report saved to: {html_path}")
    
    # Try to generate PDF
    try:
        from weasyprint import HTML as WeasyHTML
        pdf_path = output_dir / "screening_report.pdf"
        WeasyHTML(string=html).write_pdf(pdf_path)
        print(f"PDF report saved to: {pdf_path}")
    except ImportError:
        print("Note: Install weasyprint for PDF generation: pip install weasyprint")
    
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Generate screening report")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions parquet")
    parser.add_argument("--output-dir", type=str, default="data/reports", help="Output directory")
    parser.add_argument("--dataset-name", type=str, default="Cathode Materials", help="Dataset name for header")
    parser.add_argument("--max-rows", type=int, default=100, help="Max rows in table")
    args = parser.parse_args()
    
    generate_report(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
