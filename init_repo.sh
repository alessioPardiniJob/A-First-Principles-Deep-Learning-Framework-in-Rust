#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="https://github.com/alessioPardiniJob/A-First-Principles-Deep-Learning-Framework-in-Rust"
BRANCH="main"

git init
git add .
git commit -m "Initial commit: project structure for a first-principles deep learning framework in Rust"
git branch -M "$BRANCH"
git remote add origin "$REMOTE_URL"
git push -u origin "$BRANCH"
