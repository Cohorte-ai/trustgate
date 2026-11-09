# Design: agent qualification (TrustGate 0.4)

Status: draft for review, 2026-11-09. Nothing here changes `certify()`.

## Why

`certify()` answers one question about one endpoint: how consistent are its answers, with a
conformal coverage guarantee. An agent holds conversations, calls tools, writes artifacts
and escalates. Whether it is safe to deploy is a different question: across the situations
it will meet, how often does it do the right thing, and does it ever do the forbidden one?

## Scope of 0.4

In: run an agent through YAML scenarios, several times each; check what it said and did;
report per suite with honest bounds; decide; bind the result to a fingerprint; plan the
budget; generate a qualification card.

Out (0.5): LLM judges and their qualification, paired comparison of two versions, trace
ingestion from OTel or Langfuse, simulated users.

## Concepts

**Target.** Anything that can play a conversation. `run(turns, role, run_index) -> Transcript`.
Two built-ins: `http` (one request per user turn, a session id kept across turns, response
fields mapped by dotted paths) and `python` (`module:function`).

**Transcript.** One `TurnObservation` per user turn (text, outcome, tool calls, citations,
cost, latency), plus artifacts (name to list of versions) and a trace id. A tool call has a
status: `executed`, `proposed` (sent for human approval), `refused`, `failed`. Fields a
target does not report are `None`, never empty: absence of evidence is not evidence.

**Scenario.** Turns, the user's role, a suite, and expectations. Positive expectations apply
to the last turn; "never" expectations apply to every turn and every artifact version.

| Expectation | Passes when | Needs |
|---|---|---|
| `outcome: [answered, escalated]` | last turn's outcome is listed | outcome |
| `text_includes`, `text_excludes` | substring checks, case-insensitive | text |
| `tools_executed`, `tools_not_executed`, `tools_not_called` | trajectory checks over all turns | tool calls |
| `cites`, `never_cites` | citation ids | citations |
| `grounded_numbers: true` | every number in the answer appears in a tool output or a user turn | tool calls with outputs |
| `artifacts`, `artifacts_exclude` | artifact names (glob) exist; forbidden text absent from every version | artifacts |

**Three statuses per run** (Verification Map): the agent's outcome (`pass` / `fail`), the
evidence (`complete` / `incomplete`: a check needed a field the target did not report), and
the evaluator (`ok` / `error`). A run passes only if all three are good. Incomplete evidence,
evaluator errors, target errors and runs skipped by the budget cap never count as passes and
are reported separately, so a broken connector cannot look like a good agent.

**Suites.** `nominal` (does the job), `escalation` (stops or hands over when it should),
`safety` (must never happen). Each has a threshold; safety has none, because any failure is
a veto.

## Statistics

The unit of evidence is the **scenario**, not the run: ten runs of one scenario are not ten
independent tests (Harness playbook, ch. 15). So:

- Each scenario runs K times (default 5). Its result is `all_pass` (pass^K indicator) and
  its pass count.
- **pass^K** for a suite = share of scenarios that passed all K runs. Lower bound: exact
  one-sided Clopper-Pearson at 95% over scenarios.
- **Run pass rate** = mean of per-scenario pass rates, with a scenario-level bootstrap
  interval. Reported, never gated on.
- **Safety, zero failures**: upper bound on the share of scenarios that would fail at least
  once in K runs, `1 - 0.05^(1/S)` over S scenarios. 0 failures in 30 scenarios supports
  "under 9.5%", not "never".

## Decisions

Per suite, then overall (worst wins):

| Decision | Rule |
|---|---|
| `reject` | any safety run failed |
| `below_threshold` | pass^K point estimate under the suite threshold |
| `insufficient_evidence` | point estimate meets the threshold but the lower bound does not; or the suite is empty; or runs were skipped, incomplete or errored |
| `approve_as_scoped` | lower bound meets the threshold, for every gated suite |

"As scoped" is part of the name on purpose: the approval holds for these scenarios, this
target and this fingerprint, nothing wider. `non_inferior` and `regression` arrive in 0.5
with paired comparison.

Consequence worth stating in the docs: with every scenario passing, approving at a 0.80
threshold needs at least 14 scenarios in the suite (`0.05^(1/14) = 0.807`). The planner
says this before any money is spent.

## Validity envelope

A fingerprint over everything that can change the result: model alias, files matching the
listed globs (prompts, skills, tool schemas, knowledge), extra key-value pairs, the scenario
files, and the TrustGate version. Stored in the report. `trustgate status` recomputes it and
reports `current` or `stale` with the components that changed.

## Budget

`trustgate plan` runs a probe (a few scenarios, once each), measures cost and latency per
run, then reports: cost of the full run at K, the largest K a budget allows, and the
minimum scenarios per suite needed to make an approval possible at its threshold. During a
run, a hard cap stops new runs once spending reaches it; skipped runs are reported.

## Outputs

`report.json` (versioned schema) and `QUALIFICATION.md`, generated from the report and the
envelope only, so the card cannot claim more than was measured.

## Compatibility

New subpackage `theaios.trustgate.agent`, new CLI commands `qualify`, `plan`, `status`. No
change to existing commands, config keys or the `certify()` API.
