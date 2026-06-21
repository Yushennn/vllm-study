---
name: trace-call-path
description: Trace a specific execution path through vLLM or LMCache source code step by step, citing exact file:line, without summarizing or skipping intermediate steps. Use when asked to trace, walk through, or follow how a specific function/request flows through the codebase.
---

When this skill is invoked, trace the requested execution path through the
codebase using the following rules. Do not deviate from them.

## Rule 1 — Never skip a step

Show every function call in the chain, in execution order — including
seemingly trivial calls like `__init__`, property accesses, or delegation
wrappers. If a function immediately delegates to another, show both. The
user is building a mental model from scratch; "obvious" steps are exactly
the ones that need to appear.

## Rule 2 — Always cite file:line

Every claim about what the code does must be backed by an exact citation:

    file_path/relative/to/repo_root.py:LINE_NUMBER

Never say "this function calls X" without giving the line where the call
appears. Never say "defined in Y" without giving the line of the definition.
Read the actual file to confirm line numbers — do not guess.

## Rule 3 — Explain branch selection

When the code branches (if/else, match/case, dict dispatch, isinstance,
registry lookup, etc.), do two things:
1. State which branch is taken in the scenario being traced.
2. State *why* — what condition is true, what value is being dispatched on,
   what the registry maps the key to. Listing all branches without saying
   which one applies is not enough.

## Rule 4 — Pause at non-obvious decision points

After each non-trivial step — a dispatch, a queue hand-off, a process
boundary, an IPC call, a lock acquisition — stop and ask:
"What do you think happens next here?"

Do this *only* when the user has not already shown they understand the
mechanism. Do not do it for mechanical plumbing that follows obviously from
what was just shown. One short question is enough; do not include a hint
or the answer.

Resume tracing as soon as the user responds, or if they say "keep going."

## Rule 5 — Distinguish documented design from code inference

When explaining *why* something is designed a certain way, explicitly label
the source of the explanation:

- **[RFC]** — the rationale comes from a documented GitHub issue, PR, or
  discussion. Search for it, cite the URL, and quote the relevant sentence.
- **[inferred from code]** — you are reading the implementation and reasoning
  about intent. Say so.

Never blend the two in the same sentence. If you find both, present them
separately. If you cannot find a documented RFC after searching, default to
[inferred from code] and say so.

## Rule 6 — Language

All explanations default to English. Only add a short Traditional Chinese
bridge sentence (prefixed "中:") when the user explicitly asks for one in
the current message. Do not add Chinese proactively.

## Trace format

Present each step in this format:

```
Step N — function_name(...)
  File: path/to/file.py:LINE
  <one or two sentences: what this function does at this point in the trace>
  → calls Step N+1
```

When crossing a process boundary (ZMQ, asyncio queue, shared memory), mark
it explicitly:

```
── PROCESS BOUNDARY (ZMQ PUSH/PULL) ──
```

At the end of each batch of steps (or when pausing per Rule 4), ask whether
to continue or whether something needs clarification before going further.
