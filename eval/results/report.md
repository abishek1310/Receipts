# Receipts — validator evaluation

Corpus: 400 real signals · 10 briefs

## Headline

| Metric | Value |
|---|---|
| **Catch rate** | **100.0%** (30/30 injected bad citations rejected) |
| **False-positive rate** | **7.69%** (10/130 valid claims wrongly rejected) |
| Diagnosis accuracy | 100.0% (caught *and* named the right failure mode) |

## Breakdown

- Injected by type: `{'hallucinated_id': 15, 'out_of_set_id': 15}`
- Caught by reason: `{'UNKNOWN_ID': 15, 'OUT_OF_SET': 15}`

## Honest failure case

Nothing got through, but the validator is **over-strict in 10 case(s)**. The clearest one is the `spf_number` block:

> SPF 50. Redness is the part people actually notice. [sig_0025]

Rejected as `UNPARSEABLE` — No valid citation tag on a claim-bearing sentence: 'SPF 50.'

This is the trade we chose. The sentence splitter errs toward splitting, and an uncited fragment fails. That costs a rewrite; the opposite error would put an uncited claim in front of a customer.

## Naturally occurring failure (from the live rejection log)

_No rejections logged yet._ Run the app or `scripts/demo_run.py` against a real model first — §11 requires the demo's rejection moment to use a genuine failure, not one engineered to fail.
