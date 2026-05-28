# TECHIN 510 Final Project
---

## Overview

The final project simulates a professional client-developer relationship. You will:

1. **Propose your own project** — define the problem, write the spec, create a revenue model, review all code, and accept (or reject) deliverables. You never write code on your own project.
2. **Develop someone else's project** — architect the system, implement it using agentic engineering (AI-first development), write tests, and deliver a working product.

All collaborations happen through GitHub — Issues, Pull Requests, and code review. 
---

## Why This Model?

**For Proposers (Client role):** A key part of software development is defining what to build, evaluating whether it was built correctly, and giving feedback that improves the product. These are the skills of a product manager, a startup founder, or anyone who hires engineers.

**For Developers (Engineer role):** Real engineering means building to someone else's spec, not your own vision. You must interpret requirements, negotiate scope, communicate progress, and respond to feedback — all while using AI tools effectively.

---

## The Two Roles

### Role 1: Proposer (Client / Product Owner)

You are the client. You define what gets built and evaluate whether it meets your standards.

**Your responsibilities:**
- Write a Project Pitch with a revenue model
- Create a detailed `SPEC.md` with user stories and acceptance criteria
- Decompose the spec into GitHub Issues with testable acceptance criteria
- Set up branch protection on your project repo (main requires 1 review)
- Review every Pull Request your developer submits
- File bug reports with reproduction steps and screenshots
- Conduct acceptance testing at each gate
- Present the problem, revenue model, and development story at Demo Day

**You never write implementation code on your own project.**

### Role 2: Developer (AI-First Freelance Engineer)

You are the engineer. You build someone else's vision using agentic engineering.

**Your responsibilities:**
- Browse project pitches and express interest
- Write an `ARCHITECTURE.md` with C4 diagram, data model, tech stack justification, and agentic engineering plan
- Set up `CLAUDE.md` and `.cursorrules` for effective AI-assisted development
- Implement features via Pull Requests, each referencing a GitHub Issue
- Use agentic engineering (Cursor, Claude Code) for all development
- Write automated tests and conduct security review
- Respond to all PR review comments and bug reports
- Present architecture and agentic engineering approach at Demo Day

**Your skill is not writing code by hand — it is orchestrating AI to produce quality code, then verifying the output.**

---

## GIX Bucks Economy

Every project operates in a simulated economy that teaches budget management, scope-cost tradeoffs, and market validation.

**See [`gix-bucks.md`](./gix-bucks.md) for full rules and worked examples.**

Quick summary:
- Every student starts with **100 GIX Bucks**
- Proposers pay developers a **negotiated development fee**
- At Demo Day, all students distribute their remaining bucks and those earned as developers as **investments** in projects they believe are viable
- **Net Profit = Investment Received - Development Fee Paid**
- Positive net profit is normalized to **bonus points**

---

## Marketplace Matching

If you are not hired by any client, or you cannot find a developer, let your instructor and TA know.

---

## Tech Stack

The tech stack is **negotiated between proposer and developer**. Some examples are given below:

| Option | When to use |
|--------|------------|
| **Next.js + Supabase** | Multi-user apps, apps needing auth, database-heavy projects |
| **Python + Streamlit** | Data-focused apps, single-user tools, rapid prototyping |
| **Custom (pre-approved)** | Other stacks require written instructor approval by end of Week 3 |

The proposer states their stack preference in the pitch. The developer may counter-propose with justification. The final choice is recorded in the `ARCHITECTURE.md`.

---

## Conflict Resolution

### Contract Terms

The `SPEC.md` + agreed GIX Bucks fee constitute the project contract. Both parties should commit to:

- **Proposer:** Review PRs within 48 hours. Provide specific, actionable feedback. Respond to developer questions within 48 hours.
- **Developer:** Submit at least one PR per 2-week period. Respond to review comments within 48 hours. Keep the proposer informed of blockers.

### Escalation Process

1. If either party is unresponsive or breaches the contract, the other creates a GitHub Issue tagged `escalation` in the project repo.
2. Instructor reviews the GitHub audit trail (PR timestamps, Issue activity, review comments) within 1 week.
3. Instructor mediates and documents the outcome.

### Grade Impact

- **Communication & Professionalism** are graded. Ghosting, persistent non-responsiveness may result in point deduction. 

---
