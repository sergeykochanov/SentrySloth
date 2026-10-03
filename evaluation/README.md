# Offline evaluation

This is a small synthetic regression suite, not a real-world vulnerability benchmark.
Its labels describe the supplied handler contracts: request fields are untrusted,
`require_admin` is the authorization check, and `delete_account` is a protected side effect.
The streaming case assumes an unbounded input stream and a promised memory limit.
These assumptions are part of the fixture, not claims about any upstream project.
No fixture code is executed.

Cases cover reordered and commented-out guards, Python indentation, moved checks,
HTTP rejection, an output containment guard, an incomplete streaming limit, and an
unresolved dependency change. Development and holdout sets are fixed in `cases.json`.
The fixtures and labels are public; this is a workflow split, not a blind benchmark.

## Prepare and scan

From an editable development install:

```bash
python scripts/prepare_evaluation.py /tmp/sentrysloth-eval
```

This creates local repositories tagged `before` and `after`, a `repos.txt` file and a
`dataset.json` with exact SHAs. Use a new output directory for each preparation.

For each repository listed in `repos.txt`, run:

```bash
sentrysloth scan /tmp/sentrysloth-eval/guard-order --from before --to after \
  --mode both -o json -f ./reports-new/guard-order.json
```

Repeat for all eight cases. Actual scans incur provider usage. Keep old/new scanner
reports in separate directories, use the same model/settings and disable repository
profile reuse when comparing versions. Re-run several times to measure model variance.
Pass boundary assumptions to the review through repository documentation when running
model experiments; the preparation script writes each case's contract to its README.

```bash
sentrysloth evaluate /tmp/sentrysloth-eval/dataset.json ./reports-new \
  --split holdout --top 10 --review-minutes 20
```

Matching requires exact repository and commit SHAs, then literal code anchors in cited
source. Old reports lacking SHAs must have their revisions resolved and audited before
adding that provenance; never guess SHAs. Duplicate runs are rejected. Unresolved labels
are excluded from precision/recall, but their findings remain visible. Missing scans count
against case recall; incomplete scans count against coverage. Precision@k ranks confidence,
then severity, then stable ID; it is not an estimate of CVE eligibility.

Anchor matching is a reproducible approximation, not semantic adjudication. Manually review
matches and add real positive, negative and unresolved cases with documented reasons.
Record maintainer outcomes separately from the technical validity label. Keep any private
reports outside this public repository. Freeze a new holdout before changing prompts.

The evaluator reports token usage and optional measured review time. It does not estimate
provider billing: prices, cached-token discounts and failed request charges are not known.
