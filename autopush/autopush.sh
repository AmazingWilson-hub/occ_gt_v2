#!/bin/bash
export GIT_DIR=/data2/t113c52027/occ_gt_v2/.git
export GIT_WORK_TREE=/data2/t113c52027/occ_gt_v2
git add -A
git diff --cached --quiet && exit 0
MSG="Auto backup $(date '+%Y-%m-%d %H:%M') | $(git diff --cached --stat | tail -1)"
git commit -m "$MSG"
git push origin clean-main:master
