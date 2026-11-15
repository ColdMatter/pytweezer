#!/usr/bin/env bash
# Deletes the remote branches that are merged into main (checked with
# `git branch -r --merged origin/main`), plus ccr-bb840753-gv9frj, which has
# the same tip as queued-rearrange-tweezers (PR #58).
#
# Before running: PRs #19 (note-suspected-bugs) and #11 (test-suite-review)
# target exp-handler and will be closed when it goes; retarget them first.
# rearrangement-optimisation-RnD is deliberately not listed.
set -euo pipefail

git push origin --delete \
  add-load-tweezers \
  argument-scannable \
  caf-laser-tuning-optimiser \
  camera-binning-gain-continuous \
  camera-experiment \
  ccr-5201d050-o0gw7x \
  ccr-5fae8856-ur1oma \
  ccr-68138dee-w7bcwe \
  ccr-925dc1fe-d76rr1 \
  ccr-93f85d7a-qr5pil \
  ccr-bb840753-gv9frj \
  ccr-c020d47d-ll9u20 \
  ccr-ef66a471-v9tis7 \
  ci-bump-actions-node24 \
  ci-pr-dependencies \
  claude/optimiser-experiment-file-org-6mtkko \
  client-data-share \
  copy-viewer-roi-as-camera-roi \
  delete-failing-rearrangement-tests \
  documentation \
  exp-handler \
  experiment-pipelines \
  experiment-recipes \
  experiments-tree-by-folder \
  fix-analysis-bugs \
  fix-copier-share-chmod \
  fix-experiments-bugs \
  fix-gui-bugs \
  fix-motmaster-driver-bugs \
  fix-poetry \
  fix-property-editor-prints \
  fix-servers-bugs \
  grafana-interface \
  ids1054b-scope \
  image-viewer \
  land-tweezer-optimisers \
  linked-motmaster \
  log-device-rpc-errors \
  mot-cloud-fit \
  motmaster-go-overhead \
  move-helper-scripts \
  ni-motmaster-holds \
  ni-test-panel \
  notebook-rig \
  optimisers-tab \
  phasemask-from-rnd \
  plain-camera-experiments \
  rb-imaging-optimiser \
  rb-motmaster-runner \
  remote-device-control \
  results-default-analyses \
  results-over-sipyco \
  revert-analysis-bug-fixes \
  rpc \
  ruff-config-and-configreader-strip \
  sipyco-state \
  slm-phasemask-experiment \
  slm-rearrangement-from-rnd \
  slm-release-sdk-on-failure \
  storage-text-encoding \
  tests-keep-logs-off-the-lab \
  timescale-db \
  trap-site-finder \
  uv-migration \
  write-tests-skill
