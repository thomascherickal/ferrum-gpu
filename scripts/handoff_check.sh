#!/usr/bin/env bash
# Hook for CLAUDE.md rule 12. Reads the context figure scripts/statusline.sh recorded and, the first
# time it reaches HANDOFF_AT percent (default 98), tells the agent to write HANDOFF.md. It fires
# once per climb: the flag is cleared when usage drops under 90 (a new session or a compaction).
#
#   usage: handoff_check.sh [HookEventName]     (default PostToolUse; also UserPromptSubmit)
#   HANDOFF_AT        threshold in percent of the context window used (default 98)
#   HANDOFF_MAX_AGE   ignore a figure older than this many seconds (default 600)
#
# Requires jq. Silent and successful whenever there is nothing to say.
set -u
event=${1:-PostToolUse}
threshold=${HANDOFF_AT:-98}
max_age=${HANDOFF_MAX_AGE:-600}

# The freshest figure among the project dir and the current dir (a worktree session may differ).
best_file=""
best_at=0
for d in "${CLAUDE_PROJECT_DIR:-}" "$PWD"; do
    [ -n "$d" ] || continue
    f="$d/.superpowers/context.json"
    [ -f "$f" ] || continue
    at=$(jq -r '.written_at // 0' "$f" 2>/dev/null || echo 0)
    [[ "$at" =~ ^[0-9]+$ ]] || continue
    if [ "$at" -gt "$best_at" ]; then
        best_at=$at
        best_file=$f
    fi
done
[ -n "$best_file" ] || exit 0

now=$(date +%s)
[ $((now - best_at)) -le "$max_age" ] || exit 0

used=$(jq -r '.context_used_percentage // empty' "$best_file" 2>/dev/null || true)
[[ "$used" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 0

flag="$(dirname "$best_file")/handoff-fired"
if awk -v u="$used" -v t="$threshold" 'BEGIN { exit !(u >= t) }'; then
    [ -f "$flag" ] && exit 0
    : > "$flag"
    msg="CONTEXT WINDOW AT ${used}% (rule 12 threshold ${threshold}%). Stop starting new work now. Write HANDOFF.md in the repository root following CLAUDE.md rule 12 (goal, state and branch, work in progress, next steps in order, open questions, commands to run first, gotchas), commit and push it together with any finished work, then stop."
    jq -n --arg e "$event" --arg m "$msg" \
        '{hookSpecificOutput: {hookEventName: $e, additionalContext: $m}}'
elif awk -v u="$used" 'BEGIN { exit !(u < 90) }'; then
    rm -f "$flag"
fi
exit 0
