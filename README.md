# RCA Runner v0.2

This patch is a corrective version built after the first real Gemini pilot.

## Why v0.2?

The first one-episode pilot exposed several experimental-design issues:

1. `episode_id` itself contained the hidden class name.
2. Flexible Gemini spent all four evidence queries paginating market history.
3. `truncated=True` was misread as causal evidence of stale/missing data.
4. Fixed-choice compact arrays did not explicitly describe their columns.
5. Fixed market/model/order/fill windows were not perfectly aligned.
6. Single-path counterfactual P&L could pull the diagnosis away from stronger audit evidence.
7. Summary files emphasized final accuracy but did not quantify provisional -> final changes.

v0.2 addresses all seven.

## Modified files

- `budgeted_rca_env.py`
- `llm_rca_runner.py`
- `decision_rca_runner.py`

New:

- `demo_runner_v02_guardrails.py`

Other provider and `.env` files remain compatible.

---

## 1. Critical label-leakage fix

Internal episode IDs look like:

```text
dev_b2001_r00_execution_delay_s1_t252
```

That must NEVER be shown to the model.

v0.2 strips `episode_id` from:

- initial observation
- contract context

and replaces it with an opaque:

```text
case_<hash>
```

Internal evaluator output still keeps the real episode ID for reproducibility.

---

## 2. Record-cap metadata is not evidence

The raw environment may internally report:

```text
truncated=True
```

This means only:

> the interface returned the maximum allowed number of records.

It does NOT mean:

- missing market data
- stale market data
- corrupted market feed
- delayed source data

v0.2 removes `truncated` from the model-visible result.

The evaluator audit log still records:

```text
interface_metadata.truncated_by_record_cap
```

for debugging.

---

## 3. No pagination-style repeat by default

Default:

```text
max_queries_per_tool = 1
```

A flexible agent cannot spend:

```text
Q1 market
Q2 market
Q3 market
Q4 market
```

merely paging through one source.

It must choose complementary evidence families.

---

## 4. Expected operating policy is shared across protocols

Both flexible and fixed-choice agents now receive:

```text
expected_hedge_policy:
  rebalance_every_steps = 5
  observation_delay_steps = 0
  execution_delay_steps = 0
```

This is the intended operational specification, not the hidden active config.

It is necessary to make under-rebalancing and timing faults identifiable.

---

## 5. Explicit evidence interpretation

Flexible runner prompt now explicitly states:

### stale observation

Compare:

```text
market.stock_price(step=t)
vs
model.observed_stock_price(decision_step=t)
```

### execution delay

Join:

```text
orders.order_id
fills.order_id
```

and compare:

```text
submitted_step
fill_step
```

### under-rebalancing

Compare model decision spacing with expected rebalance interval.

### volatility misspecification

Compare model volatility input with market-based realized-volatility evidence,
while treating realized volatility as noisy.

---

## 6. Fixed-choice schema is explicit

v0.1 compact evidence:

```text
[43, 215, 0.53662]
```

v0.2:

```text
schema:
  [order_id, submitted_step, target_position]

records:
  [43, 215, 0.53662]

comparison_guidance:
  Join to fills by order_id and compare submitted_step with fill_step.
```

This makes the task about diagnosis rather than guessing array semantics.

---

## 7. Fixed windows are aligned

For the recent operational period, v0.2 aligns:

- market
- model decisions
- orders
- fills
- positions
- P&L

around the same recent hedge-decision window.

This is especially important for stale-vs-execution diagnosis.

---

## 8. Counterfactual interpretation

The model now receives a strong rule:

> Single-path counterfactual P&L is secondary evidence.
> A correct repair can worsen realized terminal P&L on one path.
> Do not reject a causal hypothesis solely for that reason.

Direct audit evidence has priority.

The evaluator-only reference tracking metric remains separate and hidden.

---

## 9. New metrics

Both runners now save:

```text
provisional_accuracy
final_accuracy
diagnosis_changed_rate
counterfactual_help_rate
counterfactual_harm_rate
```

Per episode:

```text
provisional_correct
final_correct
final_changed_from_provisional
counterfactual_helped
counterfactual_harmed
```

This directly tests whether the counterfactual stage helps or hurts.

---

## 10. First checks

Replace the three modified `.py` files and add the demo.

Then:

```powershell
python demo_runner_v02_guardrails.py
```

Expected final line:

```text
ALL V0.2 GUARDRAILS PASSED
```

Then API-free state-machine tests:

```powershell
python llm_rca_runner.py --smoke --limit 1 --output-dir results/v02_flexible_smoke
```

```powershell
python decision_rca_runner.py --provider smoke --split dev --limit 1 --output-dir results/v02_fixed_smoke
```

---

## 11. Real Gemini re-test

Do NOT run all 90 dev episodes yet.

First repeat a small pilot.

Flexible:

```powershell
python llm_rca_runner.py --provider gemini --split dev --limit 5 --output-dir results/gemini_flexible_v02_5
```

Fixed-choice:

```powershell
python decision_rca_runner.py --provider gemini --split dev --limit 5 --output-dir results/gemini_fixed_v02_5
```

Then inspect:

- which evidence families were selected
- provisional vs final diagnosis
- whether execution timing was joined correctly
- whether counterfactual helped or harmed
- whether normal is over-selected

Only after those traces look sensible should you run the full 90 dev episodes.

---

## 12. Recommended protocol version labels

Use these exact strings in experiment logs:

```text
flexible_agent_v02
fixed_choice_v02
```

Do not mix v0.1 and v0.2 results in the same aggregate table.
