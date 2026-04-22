# FineHero ML — Limitations

Operational limits of the current model and the appeal-letter module, with inline references. For the full literature synthesis, see [`docs/literature_synthesis.md`](docs/literature_synthesis.md).

## Selective labels

The training set is restricted to tickets that went to an OATH hearing. Reported metrics (AUC 0.87, Acc 78.9%) describe performance on that self-selected population, not on all issued tickets. Tickets paid without contest, written off, or never issued are unobserved. Outcome comparisons between the model and human adjudicators on this data are therefore systematically biased (Lakkaraju et al. 2017). In NYC terms: FY2022 saw ~1.7M tickets go to hearing with a ~30% dismissal rate — that is the slice we model.

## Evaluation baseline

The original headline AUC 0.8696 came from a random 80/20 stratified split. On dispute-style problems with per-plate history features, random splits routinely overstate AUC by 0.01–0.05 because a plate can appear in both train and test (Bergmeir & Benítez 2012; Kaufman et al. 2012). The honest, time-aware chronological AUC — full Optuna-tuned CatBoost, last 20% by `issue_date` held out — is **0.8425** (best iter 629, depth 10, lr 0.079). This is the baseline all future AUC claims should be measured against. The leakage audit in [`docs/leakage_audit.md`](docs/leakage_audit.md) (200k subsample, short training) showed random-split and time-aware AUCs agreeing to within 0.0005 under that config, so the 0.027 drop at full training capacity reflects **temporal concept drift** that a fully-tuned model overfits to on random splits, not feature-level label leakage. Plate-blocked 5-fold CV (proxy grouping) reported 0.8740 ± 0.0745; the high fold variance reflects imperfect proxy grouping rather than real plate memorization. One mild caveat: the target-shuffle probe returned AUC 0.5301 (threshold 0.48–0.52), most likely reflecting residual inter-year structure plus subsample variance rather than label leakage.

## Historical enforcement bias

High-signal features — `precinct`, `violation_code × precinct`, `issuing_agency`, `license_type × state` — encode decades of discretionary enforcement. Streetsblog's precinct-level analyses document that a small number of precincts account for a disproportionate share of certain ticket types, and that placard enforcement is near-zero in some areas. "Police data measures enforcement, not ground-truth behavior" (Lum & Isaac 2016; Pierson et al. 2020). Accuracy on this data is not the same as fairness across neighborhoods or demographics.

## Fairness trade-off is unavoidable

Calibration, equal false-positive rates, and equal false-negative rates cannot be simultaneously satisfied across groups when base rates differ (Kleinberg, Mullainathan & Raghavan 2016; Chouldechova 2017). FineHero makes no explicit fairness choice today. Any future claim of "fairness" must name the criterion being optimized and accept the ones being violated (Hardt, Price & Srebro 2016).

## Modeling choice is defensible, not optimal by default

CatBoost was chosen over deep tabular methods because tree-based gradient boosting remains state-of-the-art on tabular data at roughly this sample size (Grinsztajn, Oyallon & Varoquaux 2022), and because ordered boosting mitigates target leakage in high-cardinality categoricals like `violation_code` and `precinct` (Prokhorenkova et al. 2018). This does not imply the 42-feature set or the current hyperparameter search is optimal.

## LLM appeal letters require human verification

The appeal generator in `finehero-appeal/` uses the Anthropic API to draft dispute letters. Legal LLMs hallucinate at measured rates between 17% (commercial RAG tools like Lexis+ AI) and 58–88% (general-purpose models on verifiable legal queries) (Dahl et al. 2024; Magesh et al. 2025). The generator is a drafting aid, not legal advice. User-facing copy must state this, and users must verify every cited regulation, sign code, and factual claim before submission.

## Probability threshold is a policy choice

The 40% pursue/skip threshold reflects a risk appetite — how much user time is worth how much expected fine reduction — not a neutral property of the model. CatBoost's raw probabilities are not guaranteed to be well-calibrated; this has not been verified for FineHero. Before relying on the threshold, fit a calibration curve (Platt scaling or isotonic regression) on a held-out slice and report reliability diagrams.
