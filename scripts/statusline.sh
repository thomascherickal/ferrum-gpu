#!/usr/bin/env bash
# Status-line wrapper. Claude Code pipes a JSON document (model, cwd, context window, rate limits)
# to the status-line command; a model cannot see the status line, so this records the real
# context-window usage in <project>/.superpowers/context.json for scripts/handoff_check.sh (the
# CLAUDE.md rule 12 trigger), then hands the same JSON to the user's own status-line script so the
# display is unchanged.
#
#   FERRUM_STATUSLINE_DELEGATE   script to run for the display (default ~/.claude/statusline-command.sh)
#
# Requires jq. Never fails: a status line must not break the UI.
set -u
input=$(cat)

dir=$(printf '%s' "$input" | jq -r '.workspace.project_dir // .cwd // empty' 2>/dev/null || true)
# In a git worktree the project dir and the working dir can differ; record the figure in both so
# the hook finds it from whichever directory it runs in.
cwd=$(printf '%s' "$input" | jq -r '.cwd // empty' 2>/dev/null || true)
used=$(printf '%s' "$input" | jq -r '
    .context_window as $c
    | if $c.used_percentage != null then $c.used_percentage
      elif $c.remaining_percentage != null then 100 - $c.remaining_percentage
      else empty end' 2>/dev/null || true)

if [[ "$used" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    for target in "$dir" "$cwd"; do
        [ -n "$target" ] || continue
        mkdir -p "$target/.superpowers" 2>/dev/null &&
            jq -n --argjson u "$used" --arg t "$(date +%s)" \
                '{context_used_percentage: $u, written_at: ($t | tonumber)}' \
                > "$target/.superpowers/context.json.tmp" 2>/dev/null &&
            mv "$target/.superpowers/context.json.tmp" "$target/.superpowers/context.json" 2>/dev/null
    done
fi

delegate=${FERRUM_STATUSLINE_DELEGATE:-$HOME/.claude/statusline-command.sh}
if [ -f "$delegate" ]; then
    printf '%s' "$input" | bash "$delegate"
else
    printf 'ctx %s%%' "${used:-?}"
fi
exit 0
