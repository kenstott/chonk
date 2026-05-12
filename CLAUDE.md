CRITICAL: Never add fallback values or silent error handling. Causes masked failures and incorrect embedding results.
CRITICAL: Pure Python library only. No frontend, no servers, no web frameworks.
CRITICAL: Maximum brevity. No pleasantries. No explanations unless asked. Code and facts only.
CRITICAL: Zero commentary. Output only what was explicitly requested. No observations, no opinions, no unsolicited notes.
# Brevity examples
# BAD:  "I found the bug — it's on line 42 where the null check is missing. Want me to fix it?"
# GOOD: "Line 42: missing null check. Fix?"
# BAD:  "The feature is already implemented in chonk/community/_index.py. Want me to mark it complete?"
# GOOD: "Implemented at chonk/community/_index.py:64. Mark complete?"
# BAD:  "I've updated both files to reflect the completed status of the feature."
# GOOD: [no text — the edit tool calls are the answer]
# BAD:  "Good idea! I'll add that to CLAUDE.md right away."
# GOOD: "Done."
CRITICAL: Test errors must be resolved whether preexisting or not. Never skip or ignore failing tests.
CRITICAL: Before answering any question about what existing code does, run a Grep or Read tool call first. No exceptions. Do not answer from memory.

# Requirements Tracking
On any new requirement, constraint, feature, or design decision: spawn a background haiku agent. It reads `.claude/agents/requirements-tracker.md` for format, then appends to `docs/arch/requirements.md`. Silent — skip implementation details, bugs, questions.

# Swarm Mode & Teammate Spawning
@.claude/refs/swarm-mode.md

# Project Memory
Read `CLAUDE_MEMORY.md` at the start of any session involving GPU runs, experiment state, or scoring.