# Final Deliverable Checklist

Project: **How to Fool AI**  
Date: **2026-05-26**  
Owner: **Ting1016-git**

---

## 1) Client Bugs Fixed

- [x] Bug #1: Reveal screen showed `?` placeholders (fixed)
- [x] Bug #2: Who Wrote It writer-bonus boundary bug (fixed)
- [x] Bug #3: "Your score" display lagged until full rerun (fixed)
- [x] Bug #4: In-game guidance + post-game score history UX gap (addressed)
- [x] Bug #5: AI chat budget exhaustion hurting human replies (fixed)
- [x] Bug #6: Raw IDs (`ai_0` / `human`) in visible chat and low-quality fallback output (fixed)
- [x] Bug #7: Fixed/predictable mini-game rotation (fixed)

Evidence:
- Related issue docs under `issues/`
- Corresponding code changes in `app.py`, `game/ai_agent.py`, `game/scoring.py`, `game/game_engine.py`, `prompts/templates.py`

---

## 2) Automated Tests (>= 3) + Security Review

- [x] Added/maintained automated tests (requirement: >=3)
- [x] Local test suite passing (`63 passed`)
- [x] `.env.example` present
- [x] `.streamlit/secrets.toml.example` present
- [x] `.gitignore` excludes local secret files (`.env`, `.streamlit/secrets.toml`)
- [x] No obvious leaked Anthropic API key pattern found in repo scan

Evidence:
- Test files in `tests/`
- CI workflow in `.github/workflows/ci.yml`
- Secret templates and ignore rules in `.env.example`, `.streamlit/secrets.toml.example`, `.gitignore`

---

## 3) PR Review Feedback Handling

- [x] All PR review comments have been replied to (no unresolved review threads found)
- [x] All requested changes are addressed (no remaining open PR feedback threads found)
- [x] All review threads are resolved

Evidence to attach (GitHub UI):
- PR links:
  - [x] [PR #14](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/14)
  - [x] [PR #13](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/13)
  - [x] [PR #12](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/12)
  - [x] [PR #11](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/11)
  - [x] [PR #10](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/10)
- Screenshots showing "Resolved conversation" / no pending review threads

Note:
- Verified via GitHub API query: no unresolved review threads returned on listed PRs.

---

## 4) Final PRs Merged

- [x] Feature/fix PRs merged into target branch
- [ ] Branch protection / required checks satisfied

Evidence to attach:
- [x] Merged PR links:
  - [x] [PR #14](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/14)
  - [x] [PR #13](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/13)
  - [x] [PR #12](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/12)
  - [x] [PR #11](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/11)
  - [x] [PR #10](https://github.com/GIX-Luyao/final-project-codebase-menglh20/pull/10)
- [ ] Screenshot of green checks on merged PRs

---

## 5) Peer Evaluation Completed

- [ ] Peer evaluation submitted
- [ ] Submission confirmation recorded

Evidence to attach:
- [ ] Screenshot or submission link: ______________________
- [ ] Submission timestamp: _______________________________

---

## Final Go/No-Go

- [x] All open bugs and PR feedback addressed
- [ ] Final PRs merged (with tests and security checks)  
  (Merged confirmed; branch protection / green-check screenshot evidence still pending)
- [ ] Peer evaluation completed

If any item above is unchecked, deliverable is **not fully complete** yet.
