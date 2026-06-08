# digicam2000

Make a photo, video, or audio file look and sound like it was captured on a 2-megapixel
CCD point-and-shoot around 2002.

digicam2000 is not a filter preset. It reproduces the artifacts a real early-2000s digicam
produced, and applies them in the order they happened in the imaging chain: lens, then
sensor, then in-camera processing, then JPEG or video codec. Give it a clean, high
resolution source and the result is close to a real period clip. A low-quality source just
stacks more loss on top.

![Big Buck Bunny: source, sony, mpeg_lofi, camcorder](examples/bbb.gif)

## Install

```bash
pip install git+https://github.com/alexhergomz/Digicam2000.git
# or, from a clone:  pip install .
```

This pulls numpy, pillow and typer. Video and audio also need
[ffmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe`) on your PATH. Photos work with no
system dependencies. ImageMagick is not used.

## Quickstart

```bash
digicam2000 photo.jpg                          # writes photo.digicam.jpg
digicam2000 photo.jpg out.jpg -p kodak         # warm, saturated Kodak look
digicam2000 clip.mp4   -p camcorder -d 2002-07-04   # camcorder look with a date stamp
digicam2000 song.wav   -p digicam              # audio only
digicam2000 --list                             # show every preset
digicam2000 --help
```

The input type is detected from the extension. Default output is `<input>.digicam.<ext>`.
Useful flags: `-p/--preset`, `-s/--strength 0..1.5` (photo), `--mp` (target megapixels),
`-d/--datestamp`, `--no-audio`, `--barrel`.

## Presets

Photo camera profiles, based on the documented look of each brand:

| preset | look |
| --- | --- |
| `digicam` (default) | typical 2 MP CCD: warm, punchy, balanced |
| `kodak` | warm, very saturated, strong reds |
| `sony` | neutral to cool, contrasty, heavily sharpened |
| `canon` | clean, slightly warm, balanced |
| `nikon` | crisp, slightly cool or green |
| `fuji` | vivid, smooth highlight roll-off |

Photo scene modes: `daylight`, `flash` (hot center, dark falloff), `lofi` (noisy high ISO),
`camcorder` (low-res still grab).

Video: `digicam` (MJPEG 640x480 movie mode with IMA-ADPCM audio), `sony` (Cyber-shot
320x240 MPEG movie), `camcorder` (interlaced MiniDV 720x480 with low-light grain),
`mpeg_lofi` (macroblocked 320x240 MPEG-4). Each degrades the soundtrack too.

## Matching a real camera (framing)

digicam2000 changes the look, not the field of view (it only adds a small barrel distortion
and a 4:3 crop), so framing is up to you when you shoot. Most early-2000s digicams were 4:3
with a small sensor and a zoom that started near 35mm (35mm-equivalent) at the wide end and
reached about 105mm at full zoom, around f/2.8 to f/4.9. The small sensor kept almost
everything in focus, which phone cameras already match.

Pick a focal length close to the camera you want to imitate:

| you want | 35mm-equivalent | on an iPhone |
| --- | --- | --- |
| classic digicam, wide end | 35 to 38 mm | the 35mm framing option, or the main lens cropped a little |
| normal or portrait digicam | about 50 mm | 2x (roughly 48 to 52 mm) |
| longer zoom | 85 to 105 mm | the telephoto lens, or 2x plus a crop |

Skip the 0.5x ultra-wide (about 13 mm); no digicam was that wide. The iPhone main lens is
about 24 to 26 mm, slightly wider than a classic digicam, so the 35mm option on recent
models (or a small crop) lands closest.

## How it works

Each artifact maps to a real cause and runs at the right point in the chain:

| stage | cause | effect |
| --- | --- | --- |
| Lens | cheap zoom optics, lateral CA, light falloff | mild barrel distortion, radial R/B fringing, vignetting |
| CCD highlights | low dynamic range, halation | two-scale bloom and a soft highlight roll-off |
| CCD smear | a saturated photosite leaks charge down its column | the vertical purple streak |
| Bayer CFA | one color sampled per pixel, then interpolated | softening and false-color zipper on edges |
| Noise | shot noise (sigma proportional to sqrt of signal) plus a read floor | signal-dependent grain, chroma blotches worst in shadows |
| In-camera ISP | weak auto white balance, punchy matrix, sharpening, chroma NR | color cast, oversaturation, edge halos, color smear |
| JPEG / codec | low-quality 4:2:0, period video codecs | 8x8 blocking, chroma bleed, MJPEG / DV / MPEG |
| Audio (video) | tiny mic, cheap ADC, automatic gain control | mono, band-limit, AGC pumping, hiss, bit-crush |

Details that keep it from looking like a modern clip with a filter:

- Order is physical. Photo: optics (CA, distortion, vignette), then sensor sampling, then
  bloom, smear, clipping and noise, then demosaic, then white balance, then tone and
  sharpening, then JPEG. Light-physics steps run in linear light; tone, color and sharpening
  run display-referred, like a real ISP.
- Chromatic aberration is radial and small: zero at the center, a couple of pixels at the
  corners. It is not a uniform whole-frame color shift.
- Bloom and smear are found by brightness and contrast (local prominence), so a lamp or a
  lit window glows and smears even in a dark scene, while a flat bright sky does not.
- Video gets a 4:3 crop (a 4:3 sensor cropped the field of view, it did not squish 16:9),
  motion blur for low frame-rate capture, a limited dynamic range curve, and lens softness.
  The camcorder profile lifts shadows into low-light grain instead of clean black.
- Audio runs in capture order. Mic self-noise is added before the AGC, so the gain control
  pumps the noise floor up in quiet passages. The ADC bit-crush and sample rate come after.

References behind the model: CCD color and highlight roll-off
([smear](https://www.dpreview.com/forums/thread/2848955),
[purple fringing vs CA](https://www.dpreview.com/forums/threads/purple-fringing-vs-chromatic-aberration.1648620/)),
CA magnitude ([Imatest](https://www.imatest.com/docs/sfr_chromatic/)),
[chroma subsampling](https://en.wikipedia.org/wiki/Chroma_subsampling).

## Examples

Source on the left, then `sony`, `mpeg_lofi`, `camcorder`:

![Tears of Steel comparison](examples/tos.gif)

The GIFs are silent. Here are the rendered clips with their degraded audio (camcorder
profile, with a date stamp):

<video src="https://raw.githubusercontent.com/alexhergomz/Digicam2000/main/examples/bbb.camcorder.mp4" controls width="420"></video>
<video src="https://raw.githubusercontent.com/alexhergomz/Digicam2000/main/examples/tos.camcorder.mp4" controls width="420"></video>

If the players do not load in your browser, the files are in
[`examples/`](examples/): `bbb.{sony,mpeg_lofi,camcorder}.mp4`,
`tos.{sony,mpeg_lofi,camcorder}.mp4`.

### Audio

digicam2000 degrades sound on its own through the same mic chain (mono, hiss, band-limit,
AGC pumping, ADC bit-crush). Spectrograms of a public-domain Beethoven recording, original
then `digicam` then `camcorder`:

![audio spectrograms](examples/audio/spectrogram.png)

<audio src="https://raw.githubusercontent.com/alexhergomz/Digicam2000/main/examples/audio/piano.digicam.mp3" controls></audio>

Clips: [original](examples/audio/piano.original.mp3),
[digicam](examples/audio/piano.digicam.mp3),
[camcorder](examples/audio/piano.camcorder.mp3),
[sony](examples/audio/piano.sony.mp3).

### Reproduce

`bash examples/make_examples.sh` rebuilds the video examples from the CC BY 3.0 Blender
films (fetched on the fly; see [`examples/CREDITS.md`](examples/CREDITS.md)).
`bash test/run.sh` downloads public test data (Kodak suite, Wikimedia clips) and renders the
development validation montages.

## License

[MIT](LICENSE). The example video is CC BY 3.0, copyright Blender Foundation. The audio
example is public domain. See [`examples/CREDITS.md`](examples/CREDITS.md).
