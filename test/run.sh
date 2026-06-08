#!/usr/bin/env bash
# Reproducible test harness for digicam2000.
# Downloads lossless Kodak test images + makes a lossless test clip, runs every
# preset, and builds before/after comparison montages in test/out.
set -e
cd "$(dirname "$0")/.."
DC="${DIGICAM2000:-digicam2000}"   # installed CLI (pip install . first)
mkdir -p test/src test/out

# 1. Lossless test images (Kodak suite, 24-bit PNG) ------------------------- #
for n in 04 07 19 23; do
  [ -f "test/src/kodim$n.png" ] || curl -s -o "test/src/kodim$n.png" \
    "https://r0k.us/graphics/kodak/kodak/kodim$n.png"
done

# 1b. Sun-in-frame sunset photos from Wikimedia Commons (light-source test) - #
UA="digicam2000-test/1.0 (alexhego55@gmail.com)"
declare -A SUN=(
  [sun_to_sea.jpg]="thumb/5/59/Sun_to_sea.jpg/3840px-Sun_to_sea.jpg"
  [sun_sea_9714.jpg]="thumb/b/bc/Sunset_at_Sea_9714.jpg/3840px-Sunset_at_Sea_9714.jpg"
  [red_sea_sunrise.jpg]="9/94/Red_Sea_Sunrise_BWP.jpg"
)
for name in "${!SUN[@]}"; do
  [ -f "test/src/$name" ] || curl -s -A "$UA" -o "test/src/$name" \
    "https://upload.wikimedia.org/wikipedia/commons/${SUN[$name]}"
done

# 2. Synthetic blown-highlight still (exercises bloom + CCD smear) ----------- #
python3 - <<'PY'
from PIL import Image, ImageDraw
im=Image.new("RGB",(480,360),(18,22,30)); d=ImageDraw.Draw(im)
d.ellipse([210,60,270,120],fill=(255,255,255))
d.rectangle([60,200,180,300],fill=(200,40,40)); d.rectangle([300,200,420,300],fill=(40,120,200))
im.save("test/src/synthetic_lamp.png")
PY

# 3. Lossless test clip (no audio) ------------------------------------------ #
[ -f test/src/test_clip.mkv ] || ffmpeg -y -f lavfi -i \
  testsrc2=size=1280x720:rate=30:duration=3 -pix_fmt yuv444p -c:v ffv1 \
  test/src/test_clip.mkv 2>/dev/null

# 3b. Audio test sources --------------------------------------------------- #
# Primary: a HIGH-FIDELITY CC piano recording (44.1k stereo) paired with video,
# so the band-limiting/hiss is obvious (a lo-fi source would hide it).
if [ ! -f test/src/piano_clip.mkv ]; then
  curl -s -A "$UA" -o test/src/piano.ogg \
    "https://upload.wikimedia.org/wikipedia/commons/f/fe/Beethoven_-_Piano_Sonata_No._28_in_A_Major%2C_Op._101_-_I._Etwas_lebhaft%2C_und_mit_der_innigsten_Empfindung.ogg"
  ffmpeg -y -f lavfi -i "testsrc2=s=1280x720:r=30:d=12" -ss 12 -t 12 -i test/src/piano.ogg \
    -c:v ffv1 -c:a pcm_s16le -ac 2 -ar 44100 -shortest test/src/piano_clip.mkv 2>/dev/null
fi
# Secondary: a real CC speech video (already lo-fi 11k mono — a real-world case).
if [ ! -f test/src/test_clip_audio.mkv ]; then
  curl -s -A "$UA" -o test/src/speech_src.ogv \
    "https://upload.wikimedia.org/wikipedia/commons/f/f6/Demonstration_data_retention_at_BMVIT_speech_Hufsky_2_2007-06-07.ogv"
  ffmpeg -y -ss 5 -t 8 -i test/src/speech_src.ogv -c:v ffv1 -c:a pcm_s16le test/src/test_clip_audio.mkv 2>/dev/null
fi
# Grid for chromatic-aberration verification (center clean, corners fringe).
[ -f test/src/grid.png ] || python3 - <<'PY'
from PIL import Image, ImageDraw
im=Image.new("RGB",(1600,1200),(245,245,245)); d=ImageDraw.Draw(im)
for x in range(0,1600,40): d.line([(x,0),(x,1200)],fill=(10,10,10),width=2)
for y in range(0,1200,40): d.line([(0,y),(1600,y)],fill=(10,10,10),width=2)
im.save("test/src/grid.png")
PY
# pink(2s)|silence(2s)|1kHz(2s)|silence(1s)|pink(2s): gaps reveal AGC pumping, pink reveals band-limit
[ -f test/src/synth_audio.mkv ] || ffmpeg -y -f lavfi -i "testsrc2=s=640x480:r=30:d=9" -filter_complex \
"anoisesrc=color=pink:amplitude=0.6:sample_rate=48000:duration=2[p1];\
aevalsrc=exprs=0:s=48000:d=2[s1];\
sine=f=1000:r=48000:d=2,volume=0.5[t1];\
aevalsrc=exprs=0:s=48000:d=1[s2];\
anoisesrc=color=pink:amplitude=0.6:sample_rate=48000:duration=2[p2];\
[p1][s1][t1][s2][p2]concat=n=5:v=0:a=1[a]" \
  -map 0:v -map "[a]" -c:v ffv1 -c:a pcm_s16le -t 9 test/src/synth_audio.mkv 2>/dev/null

# 4. Run every preset ------------------------------------------------------- #
for f in test/src/kodim*.png; do b=$(basename "$f" .png)
  for p in daylight flash lofi camcorder; do
    "$DC" "$f" "test/out/$b.$p.jpg" --preset "$p" --datestamp 2002-07-04
  done
done
"$DC" test/src/synthetic_lamp.png test/out/synthetic.daylight.jpg --preset daylight
"$DC" test/src/synthetic_lamp.png test/out/synthetic.lofi.jpg --preset lofi
# Camera profiles on a portrait + a colorful subject
for cam in digicam kodak sony canon nikon fuji; do
  "$DC" test/src/kodim04.png "test/out/cam_$cam.jpg"  --preset "$cam"
  "$DC" test/src/kodim23.png "test/out/cam23_$cam.jpg" --preset "$cam"
done
python3 - <<'PY'
from PIL import Image, ImageDraw
cams=["digicam","kodak","sony","canon","nikon","fuji"]
def lab(im,t):
    im=im.copy(); d=ImageDraw.Draw(im); d.rectangle([0,0,len(t)*8+8,18],fill=(0,0,0)); d.text((4,3),t,fill=(255,255,0)); return im
def grid(prefix, src, size):
    tiles=[lab(Image.open("test/src/"+src).convert("RGB").resize(size),"ORIGINAL")]
    for c in cams:
        tiles.append(lab(Image.open(f"test/out/{prefix}{c}.jpg").convert("RGB").resize(size),c))
    cols=4; rows=(len(tiles)+cols-1)//cols; w,h=size
    g=Image.new("RGB",(w*cols+10*(cols-1), h*rows+10*(rows-1)),(20,20,20))
    for i,t in enumerate(tiles): g.paste(t,((i%cols)*(w+10),(i//cols)*(h+10)))
    return g
grid("cam_","kodim04.png",(340,510)).save("test/compare_cameras_portrait.png")
grid("cam23_","kodim23.png",(340,227)).save("test/compare_cameras_parrots.png")
print("saved camera comparison montages")
PY
for s in sun_to_sea sun_sea_9714 red_sea_sunrise; do
  [ -f "test/src/$s.jpg" ] && "$DC" "test/src/$s.jpg" "test/out/$s.daylight.jpg" --preset daylight
done
for vp in digicam_video camcorder mpeg_lofi; do
  "$DC" test/src/test_clip.mkv "test/out/clip.$vp.avi" --preset "$vp" --datestamp 2002-07-04
done

# 4b. Chromatic-aberration check (radial: clean center, fringe at corners) --- #
"$DC" test/src/grid.png test/out/grid.daylight.jpg --preset daylight
[ -f test/src/grid.mkv ] || ffmpeg -y -loop 1 -i test/src/grid.png -t 1 -r 15 -c:v ffv1 test/src/grid.mkv 2>/dev/null
"$DC" test/src/grid.mkv test/out/grid.digicam_video.avi --preset digicam_video --no-audio

# 5. Audio degradation (hi-fi piano + real speech + synthetic) + montages ---- #
for src in piano_clip synth_audio test_clip_audio; do
  [ -f "test/src/$src.mkv" ] || continue
  for vp in digicam_video camcorder mpeg_lofi; do
    "$DC" "test/src/$src.mkv" "test/out/$src.$vp.avi" --preset "$vp"
  done
done
# Piano spectrogram montage (band-limiting on a clean hi-fi source)
if [ -f test/out/piano_clip.digicam_video.avi ]; then
  psg(){ ffmpeg -y -i "$1" -lavfi "showspectrumpic=s=680x240:legend=1:scale=log" "$2" 2>/dev/null; }
  psg test/src/piano_clip.mkv           test/out/_psg_src.png
  psg test/out/piano_clip.digicam_video.avi test/out/_psg_digi.png
  psg test/out/piano_clip.camcorder.avi test/out/_psg_cam.png
  psg test/out/piano_clip.mpeg_lofi.avi test/out/_psg_mpeg.png
  python3 - <<'PY'
from PIL import Image, ImageDraw
def lab(p,t):
    im=Image.open(p).convert("RGB"); d=ImageDraw.Draw(im)
    d.rectangle([0,0,len(t)*7+8,16],fill=(0,0,0)); d.text((4,3),t,fill=(255,255,0)); return im
def vcat(rs,g=6):
    w=max(r.width for r in rs); h=sum(r.height for r in rs)+g*(len(rs)-1)
    o=Image.new("RGB",(w,h),(15,15,15)); y=0
    for r in rs: o.paste(r,(0,y)); y+=r.height+g
    return o
vcat([lab("test/out/_psg_src.png","PIANO source 44.1k stereo (full band to 22kHz)"),
      lab("test/out/_psg_digi.png","digicam_video: 11k, band ~250-5.5kHz + hiss"),
      lab("test/out/_psg_cam.png","camcorder: 32k, band ~90-13kHz"),
      lab("test/out/_psg_mpeg.png","mpeg_lofi: 22k, band ~200-8kHz, MP3")]
     ).save("test/compare_audio_piano.png")
print("saved compare_audio_piano.png")
PY
fi
if [ -f test/out/synth_audio.digicam_video.avi ]; then
  sg(){ ffmpeg -y -i "$1" -lavfi "showspectrumpic=s=640x260:legend=1:scale=log" "$2" 2>/dev/null; }
  sg test/src/synth_audio.mkv               test/out/_sg_src.png
  sg test/out/synth_audio.digicam_video.avi test/out/_sg_digi.png
  sg test/out/synth_audio.camcorder.avi     test/out/_sg_cam.png
  ffmpeg -y -i test/src/synth_audio.mkv -lavfi showwavespic=s=640x130:colors=cyan test/out/_wf_src.png 2>/dev/null
  ffmpeg -y -i test/out/synth_audio.digicam_video.avi -lavfi showwavespic=s=640x130:colors=cyan test/out/_wf_digi.png 2>/dev/null
  python3 - <<'PY'
from PIL import Image, ImageDraw
def lab(p,t):
    im=Image.open(p).convert("RGB"); d=ImageDraw.Draw(im)
    d.rectangle([0,0,len(t)*7+8,16],fill=(0,0,0)); d.text((4,3),t,fill=(255,255,0)); return im
def vcat(rs,g=6):
    w=max(r.width for r in rs); h=sum(r.height for r in rs)+g*(len(rs)-1)
    o=Image.new("RGB",(w,h),(15,15,15)); y=0
    for r in rs: o.paste(r,(0,y)); y+=r.height+g
    return o
vcat([lab("test/out/_sg_src.png","SPECTRUM source 48k (full band, silence gaps, 1kHz tone)"),
      lab("test/out/_sg_digi.png","digicam_video: band ~250-5500Hz, hiss in gaps, bitcrush"),
      lab("test/out/_sg_cam.png","camcorder: band ~90-13kHz, cleaner"),
      lab("test/out/_wf_src.png","WAVEFORM source: real silence gaps"),
      lab("test/out/_wf_digi.png","WAVEFORM digicam: gaps filled by AGC-pumped hiss")]
     ).save("test/compare_audio.png")
print("saved test/compare_audio.png")
PY
fi
echo "done -> see test/out/"
