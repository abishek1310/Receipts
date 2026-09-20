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

- Reason: `UNPARSEABLE`
- Offending IDs: `—`
- Insight: `ins_02`, attempt 1

Model output, verbatim:

> Designed specifically for outdoor and endurance athletes aged 20-40 who refuse to let their sun care compromise their performance. While traditional sunscreens run into the eyes, feel heavy and greasy, and require constant, messy reapplication during activity, this formula wins by delivering high SPF 50+ protection in an ultra-lightweight, fast-absorbing gel that resists sweat and never stings [sig_0073, sig_0116, sig_0265, sig_0277].

No valid citation tag on a claim-bearing sentence: 'Designed specifically for outdoor and endurance athletes aged 20-40 who refuse to let t...'
