\documentclass{article}

% Recommended, but optional, packages for figures and better typesetting:
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subfigure}
\usepackage{booktabs} % for professional tables

% hyperref makes hyperlinks in the resulting PDF.
% If your build breaks (sometimes temporarily if a hyperlink spans a page)
% please comment out the following usepackage line and replace
% \usepackage{icml2025} with \usepackage[nohyperref]{icml2025} above.
\usepackage{hyperref}


% Attempt to make hyperref and algorithmic work together better:
\newcommand{\theHalgorithm}{\arabic{algorithm}}

% Use the following line for the initial blind version submitted for review:
% \usepackage{icml2025}

% If accepted, instead use the following line for the camera-ready submission:
\usepackage[accepted]{icml2025}

\makeatletter
\renewcommand{\printAffiliationsAndNotice}[1]{}
\makeatother

% For theorems and such
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{amsthm}
\usepackage{enumitem}

% if you use cleveref..
\usepackage[capitalize,noabbrev]{cleveref}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THEOREMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{assumption}[theorem]{Assumption}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

% Todonotes is useful during development; simply uncomment the next line
%    and comment out the line below the next line to turn off comments
%\usepackage[disable,textsize=tiny]{todonotes}
\usepackage[textsize=tiny]{todonotes}



% The \icmltitle you define below is probably too long as a header.
% Therefore, a short form for the running title is supplied here:
\icmltitlerunning{AutoLift: An Agentic Pipeline for Iterative Uplift Model Experimentation}

\begin{document}

\twocolumn[
\icmltitle{AutoLift: An Agentic Pipeline \\ for Iterative Uplift Model Experimentation}

% It is OKAY to include author information, even for blind
% submissions: the style file will automatically remove it for you
% unless you've provided the [accepted] option to the icml2025
% package.

% List of affiliations: The first argument should be a (short)
% identifier you will use later to specify author affiliations
% Academic affiliations should list Department, University, City, Region, Country
% Industry affiliations should list Company, City, Region, Country

% You can specify symbols; they are numbered in order.
% Ideally, you should not use this facility. Affiliations will be numbered
% in order of appearance, and this is the preferred way.
\icmlsetsymbol{equal}{*}

\begin{icmlauthorlist}
\icmlauthor{Erica Chi Yi Tung (A0318639R)}{equal,NUS}
\icmlauthor{He Zhaoyu, Joey (A0318150N)}{equal,NUS}
\icmlauthor{Hrithik Kannan Krishnan (A0318899B)}{equal,NUS}
\icmlauthor{Liow Zhi Xin, Sherlyn (A0172191J)}{equal,NUS}
\icmlauthor{Zhang Jiachen, Carson (A0318512L)}{equal,NUS}
\end{icmlauthorlist}

\icmlaffiliation{NUS}{National University of Singapore, Singapore}

\icmlcorrespondingauthor{Erica Chi Yi Tung}{e1520499@u.nus.edu}
\icmlcorrespondingauthor{He Zhaoyu, Joey}{e1520010@u.nus.edu}
\icmlcorrespondingauthor{Hrithik Kannan Krishnan}{e1520759@u.nus.edu}
\icmlcorrespondingauthor{Liow Zhi Xin, Sherlyn}{e0201761@u.nus.edu}
\icmlcorrespondingauthor{Zhang Jiachen, Carson}{e1520372@u.nus.edu}

% You may provide any keywords that you
% find helpful for describing your paper; these are used to populate
% the "keywords" metadata in the PDF, but will not be shown in the document
\icmlkeywords{Machine Learning, Uplift Modelling, Causal Inference, Agentic Systems, AutoML}
\vskip 0.3in
]

% this must go after the closing bracket ] following \twocolumn[ ...

% This command actually creates the footnote in the first column
% listing the affiliations and the copyright notice.
% The command takes one argument, which is text to display at the start of the footnote.
% The \icmlEqualContribution command is standard text for equal contribution.
% Remove it (just {}) if you do not need this facility.


\begin{abstract}
Manual machine learning experimentation is iterative, time-intensive, and difficult to reproduce consistently. We present AutoLift, an agentic machine learning framework that automates the full closed-loop workflow, from data ingestion and feature engineering to model selection, hyperparameter refinement, and evaluation. Our system comprises 13 specialised agents organised across five phases, coordinated through a persistent JSONL knowledge base that accumulates trial history across runs. Unlike conventional hyperparameter search methods, agents explicitly form, test, and adjudicate hypotheses about which features and uplift strategies are likely to improve performance, directing subsequent trials toward more promising configurations. A key design principle is that LLM narrative is bounded by deterministic metric evidence: a verdict ceiling computed from the Qini AUC prevents the language model from reporting a more optimistic outcome than the data supports. Applied to the X5 RetailHero uplift modelling task, the framework evaluates four uplift learner families across 7 trials, spanning warmup and optimisation phases. The agent champion (Class Transformation with Gradient Boosting) achieves a normalised Qini AUC of 0.2540 and Uplift@10\% of 0.0518, compared to a hand-tuned manual baseline of 0.2469 and 0.0629 respectively, while eliminating the need for manual experiment scheduling. The code is available at \url{https://github.com/YOUR-REPO-LINK-HERE}.
\end{abstract}


\section{Introduction}

\subsection{Problem Statement and Motivation}
Targeted marketing campaigns commonly rely on response models that predict the likelihood of a customer making a purchase. However, these models fail to distinguish between customers who buy because of a promotion and those who would have bought regardless. This leads to budget being wasted while missing the persuadable customers that generate true incremental revenue.

Our project addresses the question: given a promotional action, which customers exhibit the largest causal lift in purchase probability? Formally, we aim to estimate the Individual Treatment Effect (ITE), or Conditional Average Treatment Effect (CATE), which captures the individual-level difference in purchase probability between treated and untreated conditions across a large retail customer base.

\subsection{Business Case and Domain Applicability}
Promotional campaigns carry a direct cost: discounts, logistics, and lost margin on customers who would have converted without an incentive. A standard response model maximises predicted purchase probability, which systematically over-targets determined customers and under-targets the persuadables. This produces two negative financial outcomes: campaign spend is diluted by customers who needed no intervention, and the truly incremental segment is incompletely reached.

In the e-commerce domain, uplift modelling shifts the goal from predicting who will buy to identifying who will buy \emph{because} of a promotion. By ranking customers on incremental response rather than raw purchase likelihood, teams can set flexible targeting thresholds, such as top 10\%, top 20\%, or within a fixed budget, to maximise return on promotional spend. At e-commerce scale, reallocating even a small share of discounts away from always-converts can translate into meaningful revenue gains.

\section{Dataset}

\subsection{X5 RetailHero Dataset Overview}
The X5 RetailHero dataset was released by X5 Retail Group as part of the RetailHero hackathon in 2019. It contains raw customer transaction histories, product information, and general customer demographics. The central problem it poses is one of causal targeting: given SMS communication, the objective is to identify which customers are genuinely persuaded to purchase as a direct result of the message, as opposed to those who would have purchased regardless or not at all.

The dataset originates from a real randomised controlled trial (RCT) conducted as part of an actual retail campaign, where treatment assignment (whether a customer received an SMS) was genuinely random. This makes it well-suited for uplift modelling, as the randomisation ensures that any observed differences in purchase behaviour between treatment and control groups can be attributed to the communication itself. The richness of the transaction history enables feature engineering across multiple time windows, which is particularly relevant to the hypothesis-driven experimentation loop at the core of this project.

\subsection{Data Collection and Access}
The X5 RetailHero dataset is accessed via the scikit-uplift Python package using the \texttt{fetch\_x5()} function, eliminating the need for manual downloads \citep{scikituplift}. The dataset comprises four files: clients, purchases, uplift train, and uplift test. The labelled training set is pre-split for uplift modelling, with a binary treatment flag and binary purchase outcome.

\section{Methodology}

\subsection{System Architecture}
\label{author info}

AutoLift is organised into five sequential phases connected by a shared persistent knowledge base. Figure~1 illustrates the data flow.

\textbf{Phase 1 — Data Understanding \& Hypothesis Setup} ingests and validates the raw data, performs a stratified 70/15/15 train/validation/test split, and constructs the initial modelling table with anti-leakage checks.

\textbf{Phase 2 — Experiment Planning} is the hypothesis engine. A Case Retrieval Agent surfaces relevant prior trial evidence; a Hypothesis Reasoning Agent converts that evidence into an actionable hypothesis; an Uplift Strategy Selection Agent picks the learner family and base estimator; and a Trial Spec Writer Agent produces a fully resolved trial specification, including feature recipe, split seed, and expected metric improvement.

\textbf{Phase 3 — Execution} validates the trial specification against a controlled vocabulary before training (Code Configuration Agent), then runs the full model lifecycle — fit, score, and artefact persistence — under a crash-recovery marker (Training Execution Agent).

\textbf{Phase 4 — Optimisation Loop} controls the search trajectory. A Refinement Agent drives a four-family warmup followed by hypothesis-guided exploration, while a Retry Controller applies three deterministic stopping rules — convergence, budget cap, and duplicate detection — before invoking any LLM reasoning.

\textbf{Phase 5 — Evaluation} adjudicates each trial. The Evaluation Judge Agent issues a verdict bounded by a deterministic metric ceiling (Section~\ref{sec:eval}), the XAI Reasoning Agent explains feature contributions via SHAP importance, and the Policy Simulation Agent translates uplift scores into business-actionable targeting decisions.

A \textbf{Persistent Knowledge Base} (JSONL ledger) connects all phases. It stores every trial record, hypothesis state, and metric in an append-only log, enabling the planning agents in Phase 2 to reason over the full experiment history on each iteration.

\subsection{Pre-processing and Feature Engineering}

\textbf{Preprocessing and feature engineering} is governed by causal ordering. Treatment is defined by \texttt{first\_issue\_date}. Any transaction on or after this date is post-treatment and excluded. \texttt{first\_redeem\_date} is fully excluded to prevent label leakage.

\textbf{Feature groups:} Features are split into six groups that can be used independently or combined:
\begin{itemize}[itemsep=2pt, parsep=0pt, topsep=0pt]
    \item \textbf{Demographic:} age (14 to 100 with invalid flag), card issue year and month, and gender.
    \item \textbf{RFM:} recency, frequency, total spend, average spend.
    \item \textbf{Basket:} total and average quantity per transaction.
    \item \textbf{Points:} loyalty points earned and spent, plus spend ratios.
    \item \textbf{Product category:} counts of products, categories, segments, brands, plus own-brand and alcohol share.
    \item \textbf{Diversity:} Shannon entropy over categories and brands.
\end{itemize}
Customers with no pre-treatment history receive zero-filled features and a no-history flag.

\textbf{Recipe-based engineering:} Features are built using six recipes: base, rfm, windowed, engagement, \texttt{product\_category}, and diversity. Each recipe selects feature groups and optional time windows, enabling feature ablation and expansion experiments. Every feature artefact is content-addressed by a SHA-256 hash of the recipe and source data fingerprint: a change in either the data or the recipe invalidates the cache and forces a full rebuild, ensuring no stale features are silently reused across runs.

\textbf{Temporal windowing:} The pipeline can compute 30, 60, or 90-day rolling features to capture recent behavioural changes that lifetime aggregates may miss.

\textbf{Anti-leakage checks:} Before saving, the feature table is validated for one row per \texttt{client\_id}, complete coverage, and the absence of treatment or target columns.

\subsection{Agent Descriptions and Responsibilities}

\subsubsection{Data Understanding \& Hypothesis Setup Phase}
\textbf{Data Ingestion Agent} loads the clients, purchases, and train tables and runs data quality checks. It verifies required columns, checks binary alignment between treatment and target, computes Average Treatment Effect as a sanity check, flags missing values above 30\%, and ensures no duplicate \texttt{client\_id} entries or broken joins. It performs a 70/15/15 stratified split on treatment and outcome to preserve class balance, and saves the resulting train, validation, and test splits as Parquet files for downstream use.

\textbf{Data Preprocessing Agent} builds the modelling table from the ingested data and prevents leakage by keeping only transactions before the \texttt{first\_issue\_date} and excluding \texttt{first\_redeem\_date}. It creates RFM and behavioural features, adds demographic attributes, fills missing history with zeros, and validates one row per client with no leakage before saving the modelling tables as Parquet files.

\subsubsection{Experiment Planning}
\textbf{Case Retrieval Agent} queries the experiment ledger to surface structured context from prior trials. It retrieves similar feature combinations, hypotheses previously supported or refuted, the best-performing uplift learner family to date, and failed runs to avoid repetition. This ensures each new trial builds on accumulated knowledge rather than repeating unsuccessful configurations.

\textbf{Hypothesis Reasoning Agent} converts retrieved evidence and current trial results into an actionable direction. It operates in one of three modes: validating an existing hypothesis when evidence supports continued exploration, refuting a hypothesis where results are contradictory, or proposing a new hypothesis where gaps in the evidence suggest an unexplored direction. Each hypothesis is assigned an action type (e.g., \texttt{recipe\_comparison}, \texttt{feature\_group\_expansion}) that is validated against the system's known action vocabulary before being persisted to the hypothesis store.

\textbf{Uplift Strategy Selection Agent} determines the full configuration for the next trial. It prioritises selecting an uplift learner family — Response Model, SoloModel, TwoModels, or ClassTransformation — before selecting the base estimator from available options including Logistic Regression, Gradient Boosting, XGBoost, LightGBM, and Random Forest. The selection is informed by prior Qini AUC, Uplift AUC, Uplift@k values, ranking stability, and the relevance of the current hypothesis.

\textbf{Trial Spec Writer Agent} translates the chosen hypothesis and strategy into a fully resolved, structured trial specification. This plan documents the hypothesis under exploration, what has changed relative to prior runs, the expected metric improvement, the exact model and hyperparameters, the selected feature recipe, split seed, and evaluation cutoff. All LLM-proposed hyperparameters are sanitised before execution: keys owned by the execution layer (e.g., \texttt{random\_state}, \texttt{seed}) are stripped, and only parameters valid for the chosen estimator are forwarded to training, preventing hallucinated fields from corrupting the model configuration.

\textbf{Feature Engineering Execution Agent} constructs the feature table as specified by the Trial Spec Writer Agent, enforcing data integrity throughout the build process and validating the output for one row per \texttt{client\_id} and the absence of target leakage.

\subsubsection{Execution Phase}
\textbf{Code Configuration Agent} consumes the trial specification and enforces a controlled vocabulary over its two critical fields before any computation. The \texttt{learner\_family} field is validated against the set \{SoloModel, TwoModels, ResponseModel, ClassTransformation\}, and the \texttt{base\_estimator} field against \{XGBoost, LightGBM, CatBoost, Logistic Regression, Random Forest\}. A \texttt{ValueError} is raised immediately on any unrecognised value, ensuring malformed or hallucinated LLM output cannot silently propagate into training.

\textbf{Training Execution Agent} manages the full model lifecycle from data preparation. As its first action it writes \texttt{status=running} to \texttt{trial\_meta.json}, establishing a crash-recovery marker distinguishable from deliberate failure. It merges the feature table, performs the stratified split, and instantiates the appropriate uplift model using the scikit-uplift library \citep{scikituplift}. Evaluation metrics are computed separately on the \emph{validation} set (used for model selection) and the \emph{held-out test} set (used as an honest independent estimate). Upon completion, artefacts written include \texttt{uplift\_scores.csv} (per-customer uplift scores, treatment indicators, and observed outcomes), serialised model pickles, and a final \texttt{trial\_meta.json} updated to \texttt{status=complete} or \texttt{status=failed}.

\subsubsection{Optimisation Loop Phase}
\textbf{Refinement Agent} uses the ledger to identify the best configuration to date and propose the next trial. In the first four successful trials it runs a warmup across all four learner families (Response Model, SoloModel, TwoModels, ClassTransformation) using Logistic Regression and the default RFM feature recipe to establish comparable baselines. After warmup, it selects the best-performing family and refines the base estimator, feature recipe, split seed, and evaluation cutoff through LLM-guided hypothesis trials. The Retry Controller blocks duplicate configurations that share the same parameter hash, learner family, and feature recipe.

\textbf{Trial Runner Agent} executes the resolved trial specification deterministically. It uses the split seed to create reproducible train, validation, and held-out sets, trains the uplift model, and saves all trial outputs including uplift scores, decile tables, Qini and uplift curves, a result card, and the trained model. A full \texttt{UpliftExperimentRecord} is appended to the ledger. If an error occurs, the failure is recorded and the pipeline continues without stopping.

\subsubsection{Evaluation Phase}
\label{sec:eval}
\textbf{Evaluation Judge Agent} receives the trial's Qini AUC, Uplift AUC, and Uplift@k metrics alongside the ledger history, then issues a structured verdict — \emph{supported}, \emph{contradicted}, or \emph{inconclusive} — on the hypothesis under test. Critically, the verdict is bounded by a \emph{deterministic metric ceiling} computed from the Qini AUC alone: a failed trial forces \emph{inconclusive} regardless of any LLM narrative; a Qini AUC at or below $-0.01$ caps the verdict at \emph{contradicted}; a Qini AUC at or above $0.05$ permits \emph{supported}; values between these thresholds admit at most \emph{inconclusive}. The LLM may propose a verdict, but it is clamped to be no more optimistic than the ceiling, ensuring that promising-sounding narratives cannot override unfavourable metric evidence.

\textbf{XAI Reasoning Agent} performs SHAP-based feature importance analysis for trained models. For SoloModel learners it computes SHAP values for the single model; for TwoModels learners it computes SHAP values for both the treatment and control models. It flags explanations that are unstable across seeds, business-unreasonable, or indicative of data leakage. When a serialised model is unavailable, the agent falls back to score-feature rank correlations (Spearman's $\rho$) to provide partial explainability without requiring model files.

\textbf{Policy Simulation Agent} converts uplift scores into targetable policy decisions, computing simulated top 5\%, 10\%, 20\%, and 30\% targeting thresholds as well as budget-constrained targeting. It reports incremental gains and coupon costs for each policy, confirming whether statistical gains translate into operational value. The recommended targeting threshold is set at the elbow of the incremental return curve rather than the tightest segment.

\subsubsection{Experiment Memory \& Retry Control Phase}
\textbf{Experiment Memory Agent} maintains an append-only JSONL ledger as the persistent knowledge base across all pipeline phases. Each record captures the \texttt{run\_id}, \texttt{parent\_run\_id}, \texttt{hypothesis\_id}, \texttt{uplift\_learner\_family}, \texttt{base\_estimator}, a \texttt{params\_hash} (SHA-256 digest of the hyperparameter dictionary), \texttt{split\_seed}, four metric fields (\texttt{qini\_auc}, \texttt{uplift\_auc}, \texttt{uplift\_at\_k}, \texttt{policy\_gain}), \texttt{verdict}, \texttt{xai\_summary}, \texttt{next\_recommended\_actions}, and \texttt{status}. Hypotheses are stored separately and tracked across six lifecycle states: \emph{proposed}, \emph{under test}, \emph{supported}, \emph{contradicted}, \emph{inconclusive}, and \emph{retired}.

\textbf{Retrieval Agent} provides a structured query interface over the ledger for use by planning agents. It supports retrieval by hypothesis status, action type, and trial linkage, and always returns the latest hypothesis state to ensure planning decisions are grounded in current evidence.

\textbf{Retry Controller Agent} applies three deterministic stopping rules before invoking any LLM reasoning. The rules are evaluated in sequence: whether the Qini AUC values of the three most recent trials fall within a convergence window; whether the cumulative trial count has exceeded the hard budget cap; and whether the candidate \texttt{params\_hash} already exists in the ledger. LLM-based reasoning is invoked only when none of the programmatic conditions trigger, returning a \texttt{RetryDecision} object containing \texttt{should\_continue}, a natural-language \texttt{reason}, and a \texttt{suggested\_next\_action} drawn from \{\textit{plan\_next\_trial}, \textit{change\_strategy}, \textit{generate\_report}\}.

\subsubsection{Output Phase}
\textbf{Manual Benchmark Agent} runs a fixed baseline trial using the TwoModel learner with Logistic Regression and the default RFM feature recipe, logging it under a dedicated \texttt{manual\_baseline} hypothesis ID. It provides a stable, human-interpretable reference point that is independent of the agent search trajectory, enabling direct comparison between automated and manually configured experimentation. The benchmark follows the same artefact schema as agent trials, returning a full metric set for consistent evaluation.

\textbf{Reporting Agent} compiles a final report once the experiment loop concludes, reading the complete knowledge base accumulated across all phases and writing the output to \texttt{logs/final\_report.md} without invoking any LLM call. It identifies the agent champion by selecting the trial with the highest normalised Qini AUC, surfaces the XAI feature importance summary from the champion trial, and reproduces the targeting cutoff alongside the full hypothesis verdict table.

\subsection{Model Registry and Safety Guardrails}

Trial configurations and results are persisted in an \texttt{UpliftLedger}, an append-only JSONL store that functions as the authoritative experiment registry. Each \texttt{UpliftExperimentRecord} is identified by a unique \texttt{run\_id} and linked to a parent run and hypothesis via pointer fields, enabling lineage tracing across the search trajectory. A \texttt{params\_hash} (SHA-256 of the hyperparameter dictionary) provides O(1) duplicate detection: when a new trial would replicate an existing configuration, the ledger returns the cached result and emits a warning rather than re-executing.

Two additional guardrails protect the execution layer from malformed LLM output. First, the Code Configuration Agent validates all LLM-proposed fields against a closed controlled vocabulary before any computation begins. Second, the \texttt{\_sanitize\_planning\_params} function strips execution-owned keys (\texttt{random\_state}, \texttt{seed}, \texttt{random\_seed}) and retains only parameters that are valid for the declared base estimator, preventing hallucinated hyperparameter names from reaching the training loop silently.

\subsection{Knowledge Base Design and Memory Management}

The knowledge base is designed around two principles: \emph{pointer-only hypothesis records} and \emph{append-only ledger writes}.

Hypothesis records (\texttt{UpliftHypothesis}) store only pointers — wave IDs and trial IDs — and never duplicate metric or policy values from the ledger. This prevents divergence between the hypothesis record and the ground-truth trial result. Metrics are always read from the ledger directly.

The JSONL ledger is append-only: existing records are never mutated. This preserves a complete audit trail of every trial, including failed runs and duplicate detections. Planning agents query the ledger at the start of each iteration to reconstruct the current state, ensuring decisions are always grounded in the most recent evidence rather than a stale in-memory snapshot.

Hypothesis lifecycle transitions are validated against an allowed-transition graph: for example, a hypothesis in \emph{under\_test} may move to \emph{supported}, \emph{contradicted}, or \emph{inconclusive}, but not directly to \emph{retired} without passing through a terminal evaluation state. Terminal states (\emph{supported}, \emph{contradicted}, \emph{retired}) are immutable once reached.

\subsection{Evaluation Metrics}
All models are evaluated using uplift-specific metrics. Standard metrics like AUC-ROC are insufficient, as they predict purchase likelihood without identifying persuadable customers.

\textbf{Qini Coefficient} measures incremental conversion gain against the proportion of the population targeted, ordered by predicted uplift. An optimal model will identify all potential converters first; a random model follows a linear baseline. The signed area between these curves, normalised by the perfect-oracle Qini, yields the normalised Qini Coefficient in $[0, 1]$.

\textbf{AUUC (Area Under the Uplift Curve)} improves on the Qini curve by using the average uplift within each decile rather than cumulative gains. It measures the expected increase in response rate when targeting a random fraction of the population based on the model's ranking.

\textbf{Uplift@k} measures the average treatment effect for the top $k\%$ of customers ranked by the model. In this work, $k \in \{10\%, 20\%\}$ reflects practical budget limits. A model with high AUUC but low Uplift@10\% performs well globally but poorly in the most actionable targeting regime. When the top-$k$ slice contains no control-group customers, the estimator is undefined and reported as \texttt{NaN} rather than zero, to avoid misleading performance claims.

\textbf{Ranking Stability Across Seeds} ensures robustness. Each experiment is run with at least two seeds, and Spearman's rank correlation measures consistency of uplift-based rankings. When rankings are unstable (low correlation) despite high AUUC, the result is treated as inconclusive.


\section{Experiments and Results}

\subsubsection{Experimental Setup}
All experiments were conducted on the X5 RetailHero uplift dataset described in Section~2. Every agent trial and the manual benchmark shared a common pre-computed feature artefact: a customer-level table of 40 features derived from demographic attributes, RFM signals, basket composition statistics, and loyalty points behaviour, built once and cached before the first trial. A 70/15/15 train, validation, and held-out split was applied to the labelled training set, with stratification on \texttt{treatment\_flg} to preserve treatment group proportions at each split boundary. Validation metrics were used for model selection; held-out test metrics are reported as independent estimates. All Qini AUC values are normalised by the perfect-oracle model Qini and are therefore directly comparable across datasets and split sizes.

A deterministic stub language model provider was used throughout all reported trials. LLM calls in the planning and evaluation phases produced context-aware responses derived from the structured input payload, making all experimental results fully reproducible without an API key. Strategy escalation from Logistic Regression to Gradient Boosting was triggered after the four-family warmup phase, and from Gradient Boosting to XGBoost and LightGBM in subsequent hypothesis trials. In total, seven agent trials and one manual benchmark trial were executed, spanning four learner families and three base estimators.

\subsection{Manual Baseline Results}

The manual benchmark used a fixed TwoModels configuration with Logistic Regression and the default RFM feature recipe. This configuration achieved a normalised Qini AUC of $0.2469$, Uplift AUC of $0.0465$, Uplift@10\% of $0.0629$, and Uplift@20\% of $0.0559$ on the validation set. This benchmark serves as the human-interpretable reference point against which the agent search trajectory is evaluated.

\subsection{Agent Insights}

Several agent behaviours warranted specific observation during the pipeline run. The \texttt{status=running} marker written by the Training Execution Agent before computation provided a lightweight fault-tolerance mechanism: any \texttt{trial\_meta.json} file found in \texttt{status=running} at pipeline restart indicates an interrupted run, allowing the retry controller to exclude it from the ledger without misclassifying it as a deliberate failure.

The SHA-256 \texttt{params\_hash} collision mechanism triggered once during the experiment. A warmup iteration proposed a configuration whose hyperparameter dictionary hashed identically to an earlier warmup trial. The Experiment Memory Agent returned the cached result and issued a warning, confirming that duplicate detection operated correctly without manual intervention.

The Retry Controller Agent terminated the search after the third hypothesis trial on metric convergence grounds, without invoking LLM reasoning. The normalised Qini values across the three hypothesis trials were $0.2540$, $0.2464$, and $0.2463$, yielding a range of $0.0077$, below the programmatic flatness threshold. All three hypothesis trials received a supported verdict from the Judge Agent; however, the deterministic verdict ceiling correctly bounded these verdicts — each trial's Qini AUC exceeded the $0.05$ supported threshold, so the ceiling permitted \emph{supported} without override. The key insight is that despite all three receiving positive verdicts, the magnitude of improvement diminished to the noise level, correctly triggering convergence.

The XAI analysis on the champion model identified \texttt{age\_clean} as the strongest driver of predicted uplift ($\rho = 0.853$ between feature value and uplift score), followed by \texttt{issue\_year} ($\rho = 0.248$) and \texttt{points\_received\_to\_purchase\_ratio\_90d} ($\rho = 0.253$). No leakage signals were detected across any of the seven trials. The consistency of \texttt{age\_clean} as the dominant driver across multiple learner families suggests a stable demographic signal for treatment heterogeneity: older customers with established purchase histories are more responsive to promotional SMS contact.

Regarding policy analysis, all four evaluated targeting cutoffs (top 5\%, 10\%, 20\%, and 30\%) produced negative ROI under the assumed cost model (coupon cost \$1 per contact, revenue per conversion \$10). The break-even lift rate under this model is $0.10$, whereas observed lift rates at tight targeting ranged from 4\% to 6\%. Positive ROI requires either a lower per-contact cost, a higher assumed conversion value, or a near-zero-cost outreach channel. The Reporting Agent correctly identified the elbow of the incremental return curve at the top-20\% cutoff and designated this as the recommended targeting threshold.

\subsection{Agent vs. Baseline Comparison}

Table~\ref{tab:results} reports the evaluation metrics for all seven agent trials and the manual benchmark. The agent champion, Class Transformation with Gradient Boosting, achieved a normalised Qini AUC of $0.2540$ and Uplift@10\% of $0.0518$. The manual benchmark achieved a Qini AUC of $0.2469$ and Uplift@10\% of $0.0629$, representing an agent improvement of $+0.0071$ normalised Qini AUC while the manual baseline leads on Uplift@10\%.

This margin should not be interpreted as a decisive performance gain. The difference falls within the range of variation observed across the warmup trials themselves, and the two primary metrics present a partially divergent story: the agent champion leads on Qini AUC while the manual benchmark leads on Uplift@10\%. The central contribution of this work is therefore not that the agent discovers a substantially superior model, but that the system autonomously explored seven configurations spanning four learner families and three base estimators, reproduced the full hypothesis testing and convergence detection cycle without manual intervention, and arrived at a result matching the manual benchmark. The automation and reproducibility of the search process is the primary value: the same pipeline can be re-executed on a new campaign dataset or feature set without any human rescheduling of experiments.

The warmup phase results further reveal that the SoloModel learner family performed substantially below the other three families (Qini AUC $0.1652$), a signal correctly identified by the Hypothesis Reasoning Agent, which did not propose returning to this family in subsequent trials. The sole meaningful performance improvement during the hypothesis phase was the transition from Logistic Regression to Gradient Boosting within the ClassTransformation family ($+0.0066$ over the warmup baseline). XGBoost and LightGBM produced no further improvement, and their near-identical scores ($0.2464$ and $0.2463$) provided the empirical convergence signal that halted the search.

\begin{table}[t]
\centering
\caption{Evaluation results for all agent trials and the manual benchmark on the
X5 RetailHero validation set. All Qini AUC values are normalised by the
perfect-oracle model. Phase labels: W = Warmup, H = Hypothesis trial.
The agent champion is shown in bold.}
\label{tab:results}
\begin{tabular}{llllrrrr}
\toprule
Phase & Learner Family & Estimator & Verdict & Qini AUC & Uplift AUC & U@10\% & U@20\% \\
\midrule
W1 & response\_model        & LogReg    & --        & 0.2380 & 0.0430 & 0.0554 & 0.0526 \\
W2 & solo\_model            & LogReg    & --        & 0.1652 & 0.0246 & 0.0182 & 0.0205 \\
W3 & two\_model             & LogReg    & --        & 0.2469 & 0.0465 & 0.0629 & 0.0559 \\
W4 & class\_transformation  & LogReg    & supported & 0.2474 & 0.0475 & 0.0566 & 0.0624 \\
\textbf{H1} & \textbf{class\_transformation} & \textbf{GradBoost}
            & \textbf{supported} & \textbf{0.2540} & \textbf{0.0436}
            & \textbf{0.0518} & \textbf{0.0651} \\
H2 & two\_model             & XGBoost   & supported & 0.2464 & 0.0441 & 0.0574 & 0.0524 \\
H3 & two\_model             & LightGBM  & supported & 0.2463 & 0.0416 & 0.0592 & 0.0549 \\
\midrule
Benchmark & two\_model      & LogReg    & --        & 0.2469 & 0.0465 & 0.0629 & 0.0559 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Knowledge Base Learning Analysis}

The knowledge base accumulated seven trial records, four warmup hypotheses, and three hypothesis-phase hypotheses across the full run. The Case Retrieval Agent successfully used ledger history to guide escalation decisions: after warmup, the best-performing family (ClassTransformation, Qini $0.2474$) was correctly identified and selected as the starting point for hypothesis trials.

Hypothesis H1 (``stronger base estimators improve uplift ranking within ClassTransformation'') was proposed based on the warmup evidence that ClassTransformation outperformed all other families at the Logistic Regression level. The hypothesis was supported at Qini $0.2540$. Hypotheses H2 and H3 tested whether XGBoost and LightGBM could improve further; both received supported verdicts from the Judge Agent (Qini AUC above the $0.05$ ceiling), but the absolute metric gains ($\Delta = 0.0076$ and $0.0077$ below H1 respectively) were negligible, and the Retry Controller correctly identified this as convergence rather than continued progress.

The SoloModel family (Qini $0.1652$, Uplift@10\% $0.0182$) was excluded from all hypothesis trials. The Hypothesis Reasoning Agent registered it as contradicted by warmup evidence and did not propose it as a candidate in any subsequent planning cycle, demonstrating that the knowledge base successfully steered the search away from unproductive regions.

\section{Advantages and Limitations}

\subsection{Advantages}

\textbf{Hypothesis-driven experimentation.} Each trial is justified by prior evidence rather than arbitrary selection. The knowledge base accumulates verdict history across trials, ensuring the search trajectory is purposeful rather than exhaustive. This reduces wasted compute on configurations that have already been explored or contradicted.

\textbf{Deterministic bounds on LLM reasoning.} The verdict ceiling mechanism prevents LLM optimism from propagating into the hypothesis store. Verdicts are bounded by the Qini AUC, meaning the system can only report a hypothesis as \emph{supported} when the metric data corroborates that claim. This makes the framework's conclusions trustworthy even when the underlying language model is prone to overconfidence.

\textbf{Reproducibility and content-addressed caching.} Feature artefacts are cached by SHA-256 hash of the recipe and source data. Duplicate trials are detected via parameter hash before any compute is initiated. The result is a fully reproducible experiment log: given the same input data and configuration, the pipeline produces identical trial sequences without re-executing any cached computation.

\textbf{Separation of validation and held-out evaluation.} Validation metrics guide model selection; held-out test metrics provide an honest independent estimate. This separation prevents the inflated metric reporting that occurs when the same data is used for both selection and evaluation.

\textbf{Modular learner and estimator registry.} Support for multiple uplift learner families (Response Model, SoloModel, TwoModels, ClassTransformation) and base estimators (Logistic Regression, Gradient Boosting, XGBoost, LightGBM, Random Forest) makes the framework dataset-agnostic. A new learner family or estimator can be registered without modifying the agent orchestration logic.

\subsection{Limitations}

\textbf{Lift magnitude insufficient for positive ROI.} The policy simulation shows negative ROI across all targeting thresholds under the standard cost model (coupon \$1, conversion value \$10). Observed lift rates of 4--6\% at tight targeting fall below the 10\% break-even threshold. The framework correctly surfaces this finding, but it implies the RetailHero SMS campaign may not be commercially viable at the current cost structure — a constraint independent of the model quality.

\textbf{Single-dataset evaluation.} All results are specific to the X5 RetailHero loyalty SMS context. Generalisation to other treatment types (email, push notifications, discount coupons), industries (banking, telecoms), or treatment assignment mechanisms requires independent validation.

\textbf{Stub LLM in reported experiments.} The published results use a deterministic stub provider rather than a live language model. The stub provides reproducibility but does not exercise the hypothesis generation quality of a real LLM. Agent decision quality with GPT-4 or Claude may differ materially from the stub's behaviour, particularly in the hypothesis proposal and strategy selection phases.

\textbf{Transaction-history feature scope.} Current features are derived solely from purchase transaction history. Behavioural signals from browsing, app engagement, or real-time contextual data could identify persuadable customers more precisely, but are not available in the RetailHero dataset.

\textbf{Binary treatment only.} The framework handles a single binary treatment flag. Multi-arm uplift settings (A/B/C testing with competing treatment conditions) or continuous dose-response estimation would require architectural extension.

\section{Conclusion}

This paper presented AutoLift, an agentic framework for iterative uplift model experimentation that combines hypothesis-driven planning, deterministic evaluation guardrails, and a persistent experiment knowledge base. Applied to the X5 RetailHero dataset, the system autonomously executed 7 trials spanning 4 learner families and 3 base estimators, reproduced the full hypothesis testing and convergence detection cycle without manual intervention, and matched the performance of a hand-tuned manual baseline (Qini AUC 0.2540 vs.\ 0.2469).

The central contribution is not a breakthrough in uplift model accuracy but the automation of the scientific method underlying ML experimentation: hypotheses are formed from ledger evidence, tested by deterministic training, and adjudicated by metrics bounded above by a ceiling that the LLM cannot override. This architecture makes the system's conclusions reproducible and auditable, regardless of which language model drives the planning phase.

The policy simulation results — negative ROI across all targeting cutoffs — demonstrate the system's ability to surface operationally relevant conclusions beyond metric rankings. In real deployment, this finding directs practitioners toward either renegotiating campaign unit economics or targeting a near-zero-cost channel before deploying an uplift model at scale.

Future work should validate the framework on higher-signal uplift datasets where incremental lift magnitude supports profitable targeting, extend hypothesis generation to multi-campaign cross-wave learning, and evaluate whether a live language model meaningfully improves upon the deterministic warmup and convergence logic demonstrated here.

The code is available at \url{https://github.com/YOUR-REPO-LINK-HERE}.

\section*{References}

\begin{thebibliography}{9}

\bibitem{scikituplift}
Maksimov, M., Yanin, I., \& Shaykhutdinov, T. (2020).
\textit{scikit-uplift: uplift modeling in scikit-learn style in Python}.
\url{https://www.uplift-modeling.com}

\bibitem{retailhero}
X5 Retail Group. (2019).
\textit{RetailHero Hackathon: Uplift Modeling Dataset}.
\url{https://ods.ai/competitions/x5-retailhero-uplift-modeling}

\bibitem{radcliffe2011}
Radcliffe, N.~J., \& Surry, P.~D. (2011).
Real-world uplift modelling with significance-based uplift trees.
\textit{White Paper TR-2011-1, Stochastic Solutions}.

\bibitem{gutierrez2017}
Gutierrez, P., \& G\'{e}rardy, J.-Y. (2017).
Causal inference and uplift modeling: A review of the literature.
\textit{Proceedings of the Machine Learning for Healthcare Conference (MLHC)}.

\bibitem{devriendt2018}
Devriendt, F., Moldovan, D., \& Verbeke, W. (2018).
A literature survey and experimental evaluation of the state-of-the-art in uplift modeling:
A stepping stone toward the development of prescriptive analytics.
\textit{Big Data}, 6(1), 13--41.

\end{thebibliography}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPENDIX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\appendix
\onecolumn
\section{Agent Architecture Summary}

The table below summarises all 13 agents, their phase assignments, whether they invoke an LLM, and their primary output artefacts.

\begin{table}[h]
\centering
\caption{AutoLift agent summary.}
\begin{tabular}{llll}
\toprule
Agent & Phase & LLM & Primary Output \\
\midrule
Data Ingestion Agent         & 1. Data Understanding   & No  & Parquet splits \\
Data Preprocessing Agent     & 1. Data Understanding   & No  & Feature table \\
Case Retrieval Agent         & 2. Planning             & No  & Prior trial context \\
Hypothesis Reasoning Agent   & 2. Planning             & Yes & Hypothesis record \\
Strategy Selection Agent     & 2. Planning             & Yes & Learner + estimator choice \\
Trial Spec Writer Agent      & 2. Planning             & Yes & Resolved trial spec \\
Feature Eng. Execution Agent & 2. Planning             & No  & Feature artefact \\
Code Configuration Agent     & 3. Execution            & No  & Validated config dict \\
Training Execution Agent     & 3. Execution            & No  & Trial artefacts + ledger record \\
Refinement Agent             & 4. Optimisation Loop    & Yes & Next trial plan \\
Trial Runner Agent           & 4. Optimisation Loop    & No  & Trial artefacts \\
Retry Controller Agent       & Memory \& Control       & Cond.& RetryDecision \\
Experiment Memory Agent      & Memory \& Control       & No  & JSONL ledger entry \\
Evaluation Judge Agent       & 5. Evaluation           & Yes & Bounded verdict \\
XAI Reasoning Agent          & 5. Evaluation           & Yes & SHAP summary \\
Policy Simulation Agent      & 5. Evaluation           & Yes & Targeting thresholds \\
Manual Benchmark Agent       & Output                  & No  & Baseline trial record \\
Reporting Agent              & Output                  & No  & Final report markdown \\
\bottomrule
\end{tabular}
\end{table}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\end{document}
