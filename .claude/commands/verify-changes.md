---
description: Comprehensive verification after code changes. Uses Boris Cherny's multi-subagent adversarial approach.
---

# Verify Changes

Run comprehensive verification of recent code changes using multi-subagent adversarial approach.

## Phase 1: Gather Context

Run these commands to understand what changed:

```bash
git diff HEAD~1 --stat
git log -3 --oneline
git diff --name-only HEAD~1
```

Identify affected files and their corresponding test files.

## Phase 2: Spawn Verification Subagents (Parallel)

Use the Task tool to spawn these subagents in parallel:

### Subagent 1: Syntax & Type Check
- Run language-specific type checker (tsc, mypy, go vet)
- Report any type errors or warnings
- Exit with list of issues found

### Subagent 2: Test Runner
- Run tests for affected files first
- Run related integration tests
- Report failures with file:line context

### Subagent 3: Lint & Style Check
- Run project linter (eslint, ruff, golangci-lint)
- Check for formatting issues
- Report violations with severity

### Subagent 4: Security Scan
- Grep for hardcoded secrets (passwords, API keys, tokens)
- Check for common vulnerabilities in new code
- Review any new dependencies

### Subagent 5: Build Validator
- Run build command
- Verify build artifacts exist
- Check for build warnings

## Phase 3: Adversarial Review

Spawn 3 adversarial subagents to review Phase 2 findings:

### Adversarial A: False Positive Filter
- Review each finding from Phase 2
- Determine if it's a real issue or false alarm
- Provide reasoning for each determination

### Adversarial B: Missing Issues Finder
- Look for issues the first pass might have missed
- Check edge cases in changed code
- Verify error handling is adequate

### Adversarial C: Context Validator
- Verify findings make sense in project context
- Check if "issues" are actually intentional patterns
- Consider project conventions

## Phase 4: Synthesize Results

Combine all subagent outputs into final report:

```
## Verification Results: [PASS/FAIL]

### Confirmed Issues
1. [Issue] - [Location] - [Why it's confirmed real]

### Warnings (Review Recommended)
1. [Warning] - [Location] - [Context]

### All Checks
- [ ] Type checking: [PASS/FAIL]
- [ ] Tests: [PASS/FAIL]
- [ ] Linting: [PASS/FAIL]
- [ ] Security: [PASS/FAIL]
- [ ] Build: [PASS/FAIL]

### Subagent Summary
- Initial findings: X issues
- After adversarial review: Y confirmed issues
- False positives filtered: Z
```

## Usage

Copy this file to your project's `.claude/commands/` directory:

```bash
cp templates/subagents/verify-changes.md .claude/commands/
```

Then invoke with: `/project:verify-changes`
