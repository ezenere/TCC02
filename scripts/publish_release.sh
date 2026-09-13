#!/usr/bin/env bash
# Publish (or update) a GitHub Release and upload assets with curl + token.
#
#   GITHUB_TOKEN=ghp_... scripts/publish_release.sh v0.2-eixo1 "Eixo 1 — treinos definitivos" \
#       runs/eixo1_resnet50_s0/checkpoints/best.pt runs/eixo1_resnet50_s0/checkpoints/last.pt ...
#
# Assets are uploaded as <run_name>__<file>, so six best.pt do not collide.
# The tag must exist locally (git tag) — the script pushes it if the remote lacks it.
set -euo pipefail

TAG="${1:?tag}"; NAME="${2:?release name}"; shift 2
[ "$#" -ge 1 ] || { echo "no assets given" >&2; exit 1; }
: "${GITHUB_TOKEN:?export GITHUB_TOKEN first}"

SLUG="$(git remote get-url origin | sed -E 's#.*github.com[:/]##; s#\.git$##')"
API="https://api.github.com/repos/$SLUG"
AUTH=(-H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json")

git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || { echo "tag $TAG does not exist locally" >&2; exit 1; }
git ls-remote --tags origin "refs/tags/$TAG" | grep -q . || git push origin "refs/tags/$TAG"

# Create the release, or reuse it if the tag already has one.
REL="$(curl -sS "${AUTH[@]}" "$API/releases/tags/$TAG" || true)"
ID="$(printf '%s' "$REL" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("id",""))' 2>/dev/null || true)"
if [ -z "$ID" ]; then
  BODY="$(python3 -c 'import json,sys; print(json.dumps({"tag_name": sys.argv[1], "name": sys.argv[2], "draft": False, "prerelease": False}))' "$TAG" "$NAME")"
  REL="$(curl -sS "${AUTH[@]}" -X POST "$API/releases" -d "$BODY")"
  ID="$(printf '%s' "$REL" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')"
  echo "release criada: $TAG (id $ID)"
else
  echo "release existente: $TAG (id $ID)"
fi
UPLOAD="https://uploads.github.com/repos/$SLUG/releases/$ID/assets"

for f in "$@"; do
  [ -f "$f" ] || { echo "asset ausente: $f" >&2; exit 1; }
  run="$(basename "$(dirname "$(dirname "$f")")")"       # runs/<run>/checkpoints/x.pt -> <run>
  asset="${run}__$(basename "$f")"
  echo "upload $asset ($(du -h "$f" | cut -f1))"
  curl -sS "${AUTH[@]}" -H "Content-Type: application/octet-stream" \
       --data-binary @"$f" "$UPLOAD?name=$asset" -o /dev/null -w "  http %{http_code}\n"
done
echo "https://github.com/$SLUG/releases/tag/$TAG"
