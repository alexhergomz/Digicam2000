#!/usr/bin/env bash
# Reproducibly build the example clips: download short excerpts of two CC BY 3.0
# Blender open movies and render each through the three "creepy" video profiles.
# (See CREDITS.md.) Source clips are fetched via HTTP range, so only ~tens of MB
# are downloaded, not the full films.
set -e
cd "$(dirname "$0")/.."
DC="${DIGICAM2000:-digicam2000}"   # the installed CLI (run `pip install .` first)
UA="Mozilla/5.0 digicam2000"
mkdir -p _srcvid examples

declare -A URL=(
  [bbb]="https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov|30"
  [tos]="https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov|600"
)
for v in bbb tos; do
  url="${URL[$v]%|*}"; start="${URL[$v]#*|}"
  [ -f "_srcvid/$v.mp4" ] || ffmpeg -y -user_agent "$UA" -ss "$start" -i "$url" -t 25 -c copy "_srcvid/$v.mp4" 2>/dev/null
  for p in sony mpeg_lofi camcorder; do
    args=(); [ "$p" = camcorder ] && args=(--datestamp 2002-07-04)
    "$DC" "_srcvid/$v.mp4" "/tmp/_ex_$v_$p.avi" --preset "$p" "${args[@]}" >/dev/null 2>&1
    ffmpeg -y -i "/tmp/_ex_$v_$p.avi" -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p \
      -aspect 4:3 -c:a aac -b:a 96k -movflags +faststart "examples/$v.$p.mp4" 2>/dev/null
    rm -f "/tmp/_ex_$v_$p.avi"
    echo "wrote examples/$v.$p.mp4"
  done
done
echo "done -> examples/"
