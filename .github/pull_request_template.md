## What problem does this solve?

<!-- One sentence: what gap does this close or what was broken? Link the issue if one exists. -->

Closes #<!-- issue number, or delete this line -->

## What changed?

<!-- Briefly describe the approach. For retrieval / citation / reasoning changes, explain how the
     groundedness contract is preserved: every answer must either cite a verified source quote or
     be explicitly labelled as reasoned extrapolation. -->

## How was it tested?

<!-- Custom Guru has two test tiers:
     - Offline suite (`pytest -q`) — fast, no LLM calls, safe on CI.
     - Live smoke-test — manual run against a real corpus + provider key (never in CI).
     Check everything that applies. -->

- [ ] `python -m pytest -q` passes (offline suite)
- [ ] Manually smoke-tested against a local corpus with a live provider (if behaviour changed)
- [ ] No live provider test needed (docs / chore / trivial refactor)

## Groundedness contract

<!-- Custom Guru's core promise: every answer either traces to a verified source quote or is
     clearly labelled as extrapolation. Confirm the boxes that apply. -->

- [ ] This change does not touch the retrieval, citation, or reasoning paths — no contract review needed
- [ ] This change touches those paths, and the groundedness contract is preserved — explanation:

  <!-- explain here if applicable -->

## Checklist

- [ ] Follows Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`)
- [ ] No secrets, `.env` files, API keys, or corpus data (`data/`) included
- [ ] New or changed behaviour is covered by tests (or explain why tests are not applicable)
- [ ] Docs / comments updated where needed
