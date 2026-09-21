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

## Where valid attribution still goes wrong

The catch rate above measures one thing: **does every cited ID resolve to a
signal that was retrieved for this generation?** That is the §1 invariant and it
holds 100% of the time.

It is not the same as "the cited signal supports this sentence". Reading real
output from the deployed app and opening every expander turned up three ways a
citation stays perfectly valid while the claim drifts away from it. All three
pass the validator. None of them are fixable by a stricter regex, because none
of them are citation-format problems.

### 1. Sentiment inversion — the citation is verbatim, the stance is reversed

Generated copy:

> "Applying occlusive recovery creams to a flushed face is like spackling
> drywall over active inflammation." `[sig_0033]`

`sig_0033`, verbatim:

> "I found Avene cicalfate **helped** calm down the redness and inflammation,
> but it feels like you're spackling dry wall on your skin."

The phrase is genuinely theirs. The person is **recommending** the product — it
worked, it just felt thick. The copy uses their words to argue the opposite.

This is the most dangerous of the three precisely because it looks like the
*strongest* citation: a vivid phrase lifted word-for-word from a real comment.
A judge who opens the expander and reads past the first clause sees a satisfied
customer quoted as a complaint.

### 2. Scope and certainty upgrade — one hedged anecdote becomes a general law

Generated copy:

> "Leaving perspiration on the skin after exercise **can trigger** fungal
> infections like tinea versicolor." `[sig_0061]`

`sig_0061`, verbatim:

> "**Could be** a fungal infection like tinea versicolor. **I** get tinea **on
> my torso** when I sweat a lot."

One person, hedged, about their torso, becomes a general causal claim about
everyone's face. The clinical vocabulary is *not* invented — r/SkincareAddiction
users write "tinea versicolor" and "closed comedones" themselves — so a
banned-words list does not catch this. What changed is scope and confidence.

This gets worse under a "make it more medical" revision, which is why that is
the one direction we would not demo.

### 3. Relevance drift — on-theme, but not about this claim

A hook about sunscreen stinging the eyes cited `sig_0066`:

> "The sweat is no joke. The gym or any exercise is fantastic... I had acne just
> like yours"

Retrieved for the brief, genuinely about sweat and exercise, and says nothing
about sunscreen or eyes. Retrieval matched the theme; the sentence needed
something narrower.

### Why we did not "fix" these

Every fix requires judging whether a passage *supports* a claim, and that
judgement needs a language model. The moment a model decides what counts as
support, the guarantee becomes "a model thinks this is fine" — which is the
thing this project exists to replace. We would rather enforce one property
completely than four properties approximately.

What we ship instead: the evidence is one click away, verbatim and linked, so a
human can catch all three in seconds. That is strictly more than a pipeline
which either prints uncited copy or prints citations nobody can check.


## Naturally occurring failure (from the live rejection log)

- Reason: `UNPARSEABLE`
- Offending IDs: `—`
- Insight: `ins_02`, attempt 1

Model output, verbatim:

> Sweating and getting wet rapidly decreases the protection time of your daily sunscreen [sig_0105]. Instead of stopping your training session to set an 80-minute reapplication alarm, you need a formula built for high endurance [sig_0073]. Try our sweat-proof sport shield today.

No valid citation tag on a claim-bearing sentence: 'Try our sweat-proof sport shield today.'
