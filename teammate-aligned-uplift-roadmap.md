So we have finalised what we gonna build on, the dataset will be the @ retailhero-uplift.zip.

Necessary Agents
I. Data Understanding + Hypothesis Setup (Joey)
Data Ingestion Agent
Loads clients, purchases, and train tables from X5 data
Validates schema and checks for nulls
Confirms treatment and target columns are present
Splits labeled customers into train/validation/test sets after confirming schema and treatment/target alignment. Feature engineering must not use target information from validation/test customers.
Checks nulls, duplicates, join keys, row counts

Data Preprocessing Agent (build outside the loop) > creates the first usable modeling table
For example:
join clients + purchases + train
aggregate one row per customer_id
create basic RFM features
validate treatment/target alignment
confirm no leakage

II. Experiment Planning → decides which models to consider (Erica)
Case Retrieval Agent
Reads prior trial history from experiment memory
Retrieves: similar feature recipes, prior supported/refuted hypotheses, best uplift learner family so far, failed runs to avoid repeating

Hypothesis Reasoning Agent
Converts retrieved evidence + current results into the next trial idea
Can do three things: validate existing hypothesis | refute existing hypothesis |  propose new hypothesis
Example: “90-day recency features improved AUUC but XAI shows instability, so refute ‘longer history is always better’ and propose ‘recent-window + dormancy interactions’”

Uplift Strategy Selection Agent
Selects uplift learner family first, not just model class
Selects the uplift strategy based on prior Qini, AUUC, Uplift@k, ranking stability, and hypothesis relevance.
Recommended shortlist: response-model baseline with XGBoost | SoloModel + XGBoost / LightGBM / CatBoost | TwoModels + XGBoost / LightGBM / CatBoost | optional ClassTransformation + tree booster
Chooses: feature recipe | learner family | base estimator | split seed | evaluation cutoff set

Trial Spec Writer Agent
Produces a structured trial plan: hypothesis being tested + what changed from previous run + expected metric improvement + exact model & params + feature recipe + stop criteria

Trial Spec Writer Agent
Converts the hypothesis into a concrete feature plan
Chooses feature windows such as 30d, 60d, 90d, lifetime
Selects feature families such as RFM, dormancy, spend trend, category diversity, basket stability

Feature Engineering Execution Agent
Builds the requested features
Validates one row per customer_id
Checks no leakage
Sends the feature table to training



III. Execution (Hrithik)
Code Editing Agent / Config / Trial Assembly Agent
- Converts the trial spec into a config file
- Selects registered model, params, registered feature recipe, and registered metric (Evaluatation metrics:
uplift scores
Qini
AUUC
Uplift@k
uplift curve
decile tables
Also stores run time and failure mode)

Initial Execution Agent
Runs one baseline trial per uplift strategy using default params from the training script provided by the code editing agent 
Suggested warm-up baselines:
1. Response model baseline + XGBoost
2. SoloModel + XGBoost or LightGBM
3. TwoModels + XGBoost or LightGBM
4. ClassTransformation + tree booster, optional

Establishes a performance floor for each algorithm
Populates the knowledge base with the first 4 trial records (from the four models included in the model selection agent) before optimization

Optimization Planning Agent
- Reads experiment memory and current champion trial
- Identifies unresolved, supported, or refuted hypotheses
- Decides whether the next trial should test:
1. a new feature recipe,
2. a new uplift learner family,
3. a new base estimator,
4. a new policy threshold,
5. or hyperparameter refinement
- Defines the Optuna search space, trial budget, and objective metric
- Avoids repeated or low-value experiments
- Produces a structured trial spec

Optuna Hyperparameter Optimizer
- Runs Bayesian optimization inside the search space defined by the agent
- Optimizes Qini, AUUC, or Uplift@k
- Returns the best parameter configuration

Uplift Training + Evaluation Agent
- Fits the selected uplift model
- Computes Qini, AUUC, Uplift@k, uplift decile tables, and policy gain
- Records training time, errors, model artifacts, and metric outputs
- Appends a trial record to experiment memory after every run

Hypothesis Verdict Agent
- Compares the new trial against the current champion
- Uses evaluation, XAI, and policy simulation results
- Labels the hypothesis as supported, refuted, or inconclusive
- Sends the result back to experiment memory for the next retry decision

IV. Experiment Memory + Retry Control
Experiment Memory Agent
Stores every run as a structured record, not just a free-text note.
Each record should include: 
run_id
parent_run_id
hypothesis_id
stage_origin
dataset version
feature recipe id
uplift learner family
base estimator
params hash
split seed
Qini
AUUC
Uplift@k
top segment summary
XAI summary
policy summary
verdict = supported / refuted / inconclusive
next recommended actions
Experiment Registry can be used by JSONL
Stores hard fields: metrics | params | feature recipe | learner family | verdict | hypothesis id
Artifact Store: config files | uplift curves | SHAP plots | decile tables | logs | markdown summaries
Retrieval Index: trial summaries | hypothesis notes | XAI findings | failure notes
Hypothesis Graph: Tracks proposed | tested | supported | refuted | superseded

Retrieval Agent
Retrieves relevant prior runs for the LLM
Should retrieve: structured metrics | trial summaries | XAI notes | refuted hypotheses

Retry Controller Agent
Decides whether to run another experiment
Retry is allowed only if: a real hypothesis is being tested | there is unexplored meaningful variation | the expected gain is not exhausted
Stop if: Qini/AUUC gains flatten | explanations stabilize | policy recommendation stabilizes | trial budget is exhausted

VII. Output
Manual Benchmark Agent
Runs a human-designed baseline pipeline
Useful for “agent vs manual” comparison

Reporting + Decision Guide Agent
Produces final outputs: champion uplift model | supported hypotheses | refuted hypotheses | key uplift drivers | targetable customer segments | recommended policy threshold | limitations and future work

