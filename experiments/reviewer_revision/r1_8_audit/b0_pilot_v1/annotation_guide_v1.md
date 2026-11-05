# R1-8 B0 Pilot Annotation Guide v1

## Purpose

This is a 60-case pilot for testing and refining the annotation scheme.
It is NOT part of the later representative false-hard-negative rate
estimate.

Reviewer A and Reviewer B annotate independently before adjudication.

## Labels

### supported_in_scope

Use this label when the visible candidate satisfies the explicit
requirements of the query within the scope that can actually be checked.

The candidate does not need to be the dataset's labeled target.
A different implementation may still be a valid alternative.

### violated

Use this label only when there is concrete evidence that the candidate
violates one or more explicit requirements in the query.

Do NOT mark a candidate violated merely because:
- it differs from the dataset target;
- it uses another implementation;
- its repository/file differs;
- it was mined as a hard negative.

State which requirement is violated and identify the supporting code or
behavioral evidence.

### unknown

Use this when the available evidence is insufficient for a reliable
supported/violated judgment.

Typical reasons:
- ambiguous query;
- missing repository context;
- external state or unavailable dependencies;
- missing caller/callee behavior;
- truncation;
- insufficient information to establish semantics.

Failure to execute because an environment or dependency is unavailable
does not itself prove a semantic violation.

## Required fields

For each case provide:

- label:
  supported_in_scope / violated / unknown

- explicit_requirements:
  What concrete behavior does the query require?

- evidence:
  What in the visible code supports the judgment?

- acceptable_alternative:
  yes / no / unclear

- dependency_issue:
  yes / no / unclear

- truncation_or_context_issue:
  yes / no / unclear

- confidence:
  high / medium / low

- notes:
  optional

## Blinding

Do not try to infer:
- hard-negative source;
- retriever/model identity;
- retrieval score;
- which candidate the system prefers.

## Independence

Reviewer A and Reviewer B must finish independently.
Do not inspect the other reviewer's labels before completing your own.

## Adjudication

After both independent reviews are complete:

1. preserve both original labels;
2. identify disagreements;
3. discuss the evidence;
4. record the final adjudicated label;
5. record the reason for adjudication.

Never overwrite an original reviewer judgment.
