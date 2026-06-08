#!/usr/bin/env python3
"""
digicam2000: give photos, video and audio an authentic early-2000s digital-camera look.

The pipeline is physically motivated: it reproduces the artifacts in the order a
real CCD point-and-shoot produced them, so a lossless / RAW-quality input comes
out looking like it was shot on a 2-4MP digicam circa 2001-2004.

Why these artifacts (the real causes):
  * Lens .......... cheap zoom -> barrel distortion, vignetting (cos^4 falloff),
                    lateral chromatic aberration (R/B focus at a different radius
                    than G -> colored fringes on high-contrast edges).
  * CCD sensor .... small pixels + low dynamic range -> highlights clip with a
                    soft bloom roll-off, shadows crush; saturated pixels leak
                    charge down their column -> the vertical purple smear.
  * Bayer CFA ..... one color per pixel, interpolated -> softening and
                    false-color "zipper" on fine edges.
  * Noise ......... photon shot noise (sigma ~ sqrt(signal)) + a read-noise floor
                    for luma, plus low-frequency chroma blotches worst in shadows.
  * In-camera ISP . weak auto white balance (warm/green cast), a punchy color
                    matrix (oversaturation, magenta-ish skin), aggressive
                    unsharp masking (edge halos), chroma noise reduction.
  * JPEG .......... moderate-to-low quality with 4:2:0 chroma subsampling ->
                    8x8 DCT blocking and chroma bleed.

Photos run through a numpy pipeline (operations done in linear light where the
physics demands it). Video is handed to ffmpeg with a matching filtergraph and a
period-correct low-bitrate codec.

Usage:
  digicam2000 in.png [out.jpg] [--preset digicam] [--mp 2.0]
  digicam2000 in.mov out.avi --preset camcorder --datestamp 2002-07-04
  digicam2000 --list

Dependencies: numpy + Pillow + typer (photos/CLI), ffmpeg/ffprobe (video).
No ImageMagick.
"""
import os, sys, subprocess, shutil, json, wave, tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

__version__ = "1.0.0"
console = Console()

# --------------------------------------------------------------------------- #
# small numpy DSP helpers (no scipy)
# --------------------------------------------------------------------------- #
def _box1d(a, r, axis):
    """Box blur of radius r along one axis via cumulative sum (edge-padded)."""
    if r < 1:
        return a
    a = np.moveaxis(a, axis, 0)
    pad = np.pad(a, [(r, r)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    cs = np.cumsum(pad, axis=0)
    cs = np.concatenate([np.zeros((1,) + cs.shape[1:], cs.dtype), cs], axis=0)
    out = (cs[2 * r + 1:] - cs[:-(2 * r + 1)]) / (2 * r + 1)
    return np.moveaxis(out, 0, axis)


def gauss_blur(a, sigma):
    """Separable Gaussian approximated by 3 box passes (spatial axes only)."""
    if sigma <= 0:
        return a
    r = max(1, int(round(sigma * 0.85)))
    for _ in range(3):
        a = _box1d(a, r, 0)
        a = _box1d(a, r, 1)
    return a


def sample_bilinear(img, mapx, mapy):
    """Bilinear resample img (H,W,C) at floating coords mapx,mapy (H,W)."""
    H, W = img.shape[:2]
    x0 = np.floor(mapx).astype(np.int64); y0 = np.floor(mapy).astype(np.int64)
    wx = (mapx - x0)[..., None]; wy = (mapy - y0)[..., None]
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x0 + 1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y0 + 1, 0, H - 1)
    Ia = img[y0c, x0c]; Ib = img[y0c, x1c]; Ic = img[y1c, x0c]; Id = img[y1c, x1c]
    return Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy) + Ic * (1 - wx) * wy + Id * wx * wy


def srgb_to_linear(x):
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def linear_to_srgb(x):
    a = 0.055
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)


# --------------------------------------------------------------------------- #
# individual degradation stages (photo)
# --------------------------------------------------------------------------- #
def lens_distort_ca(lin, k_barrel, ca):
    """Barrel distortion + lateral chromatic aberration (per-channel radial scale)."""
    if k_barrel == 0 and ca == 0:
        return lin
    H, W = lin.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    nx = (xx - cx) / cx; ny = (yy - cy) / cy
    r2 = nx * nx + ny * ny
    out = np.empty_like(lin)
    scales = (1.0 + ca, 1.0, 1.0 - ca)  # R drifts out, B drifts in
    for ch in range(3):
        f = (1.0 + k_barrel * r2) * scales[ch]
        mx = cx + nx * f * cx
        my = cy + ny * f * cy
        out[..., ch] = sample_bilinear(lin[..., ch:ch + 1], mx, my)[..., 0]
    return out


def vignette(lin, amount):
    """cos^4-style optical light falloff toward the corners."""
    if amount <= 0:
        return lin
    H, W = lin.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2)
    mask = (1.0 - amount * (r ** 2.2))
    return lin * mask[..., None]


def light_source_map(lin):
    """Estimate where the actual light sources / blown highlights are.

    Two physical cues:
      1. local prominence: a bright blob standing out against a darker surround
         (sun, lamp, specular glint). This is the key discriminator: an orange
         sunset *sky* is bright everywhere, so it has low prominence and must not
         bloom, while the sun disk sitting in it does. The local average is taken
         over a large radius so even a big sun still stands out from sky+sea.
      2. per-channel saturation: any channel near full well (a deep-orange sun
         saturates its red photosites even though blue is low), which is what
         physically triggers blooming/smear; it amplifies prominent cores.
    Returns (L, clip): L drives bloom, clip drives smear.
    """
    lum = lin.mean(2)
    H, W = lum.shape
    sig = max(8, min(H, W) // 10)                          # wide local average radius
    base = gauss_blur(lum[..., None], sig)[..., 0]
    prom = np.clip(lum - base, 0, None)
    hi = float(np.percentile(prom, 99.5))                  # robust normalize to the brightest blob
    prom = np.clip(prom / (hi + 1e-6), 0, 1)
    bright = np.clip((lum - 0.5) / 0.5, 0, 1)
    sat = np.clip((lin.max(2) - 0.9) / 0.1, 0, 1)          # any channel near full well
    core = prom * (0.4 + 0.6 * bright)                     # prominent AND bright
    L = np.clip(core * (1.0 + sat), 0, 1)                  # clipping amplifies prominent cores
    clip = np.clip(sat * (0.3 + 0.7 * prom), 0, 1)         # smear: saturated AND prominent (not flat sky)
    return L, clip


def highlight_bloom(lin, L, amount, sigma):
    """CCD halation / veiling glare, keyed to the light-source map.

    Two scales: a tight halo (sigma) plus a wide, low-amplitude veil (4*sigma),
    approximating how lens glare spreads around a bright source. The glow takes the
    source's own color, so a warm sun blooms warm.
    """
    if amount <= 0:
        return lin
    src = L[..., None] * np.clip(lin, 0, None)
    glow = gauss_blur(src, sigma) * 0.65 + gauss_blur(src, sigma * 4.0) * 0.35
    return lin + glow * amount


def ccd_smear(lin, clip, amount):
    """Vertical charge-leak streaks from *saturated* pixels (the CCD tell), in linear light.

    Physically the streak is overflow CHARGE added along the readout column, so it is
    additive, and its colour depends on how much charge there is:

      * a faint streak carries a magenta/purple cast (the green photosites leak a bit
        less than red and blue on a typical CCD), but
      * as the charge grows it floods every channel, so a bright streak desaturates and
        washes out to WHITE (it is not a saturated purple).

    We model this: `q` is the (additive) charge, with a white-hot core at the source
    fading along the column; the tint is interpolated from purple at low q to white at
    high q, so bright sources clip to white and only the dim parts stay purple.
    """
    if amount <= 0:
        return lin
    over = np.clip(clip, 0, 1) ** 1.2
    decay = 0.985                             # slow fade: a long streak that grades white core -> purple tail
    up = np.empty_like(over); acc = np.zeros_like(over[0])
    for i in range(over.shape[0]):
        acc = np.maximum(over[i], acc * decay); up[i] = acc
    dn = np.empty_like(over); acc = np.zeros_like(over[0])
    for i in range(over.shape[0] - 1, -1, -1):
        acc = np.maximum(over[i], acc * decay); dn[i] = acc
    q = np.maximum(up, dn) * amount           # additive charge profile (white-hot core, fading tail)
    t = np.clip(q / 0.55, 0, 1)[..., None]    # desaturation: 0 -> purple, 1 -> white
    purple = np.array([0.85, 0.40, 1.0], np.float32)
    color = purple * (1 - t) + t              # white is 1.0 on every channel
    return lin + q[..., None] * color


def bayer_emulate(lin, strength):
    """Mosaic to an RGGB Bayer grid then interpolate back -> CFA softening + false color."""
    if strength <= 0:
        return lin
    H, W = lin.shape[:2]
    R, G, B = lin[..., 0], lin[..., 1], lin[..., 2]
    mR = np.zeros((H, W), np.float32); mG = np.zeros((H, W), np.float32); mB = np.zeros((H, W), np.float32)
    mR[0::2, 0::2] = 1; mB[1::2, 1::2] = 1
    mG[0::2, 1::2] = 1; mG[1::2, 0::2] = 1
    mosaic = R * mR + G * mG + B * mB

    def demo(mask, sigma):
        vals = gauss_blur((mosaic * mask)[..., None], sigma)[..., 0]
        cnt = gauss_blur(mask[..., None], sigma)[..., 0]
        return vals / np.maximum(cnt, 1e-6)

    rec = np.stack([demo(mR, 0.9), demo(mG, 0.7), demo(mB, 0.9)], -1)
    return lin * (1 - strength) + rec * strength


def add_noise(lin, lum_amt, chroma_amt, rng):
    """Signal-dependent shot noise (luma) + low-frequency chroma blotches (worst in shadows)."""
    if lum_amt <= 0 and chroma_amt <= 0:
        return lin
    sig = np.clip(lin, 0, 1)
    if lum_amt > 0:
        std = lum_amt * (np.sqrt(sig) + 0.04)
        lin = lin + rng.standard_normal(lin.shape).astype(np.float32) * std
    if chroma_amt > 0:
        cn = gauss_blur(rng.standard_normal(lin.shape).astype(np.float32), 2.0)
        cn = cn - cn.mean(2, keepdims=True)  # opponent-color only
        shadow = 0.3 + 0.7 * (1 - sig.mean(2, keepdims=True))
        lin = lin + cn * chroma_amt * shadow
    return lin


def highlight_knee(lin, knee):
    """Compress the top of the range -> the gentle CCD highlight roll-off."""
    if knee >= 1.0:
        return lin
    x = lin
    over = x > knee
    x = np.where(over, knee + (1 - knee) * (1 - np.exp(-(x - knee) / (1 - knee))), x)
    return x


def s_curve(x, contrast, black):
    """Display-referred S-curve contrast + lifted/crushed black point."""
    x = np.clip((x - black) / (1 - black), 0, 1)
    if contrast != 0:
        k = contrast
        x = x + k * (x - 0.5) * (1 - np.abs(2 * x - 1))  # gentle symmetric S
    return np.clip(x, 0, 1)


def saturate(x, sat, skin_magenta):
    """Boost saturation; nudge reds toward magenta (period skin-tone tell)."""
    luma = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = luma + (x - luma) * sat
    if skin_magenta > 0:
        redness = np.clip(x[..., 0] - x[..., 2], 0, 1)[..., None]
        x = x + np.array([0, -0.4, 0.4], np.float32) * redness * skin_magenta * 0.1
    return np.clip(x, 0, 1)


def unsharp(x, amount, sigma):
    """In-camera sharpening -> visible edge halos."""
    if amount <= 0:
        return x
    blur = gauss_blur(x, sigma)
    return np.clip(x + (x - blur) * amount, 0, 1)


def chroma_nr(x, amount):
    """Camera chroma noise reduction: blur color, keep luma -> the 'watercolor' smear."""
    if amount <= 0:
        return x
    luma = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    chroma = x - luma
    chroma = chroma * (1 - amount) + gauss_blur(chroma, 1.6) * amount
    return np.clip(luma + chroma, 0, 1)


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #
PRESETS = {
    # Outdoor daylight: punchy CCD color, mild everything, warm auto-WB.
    "daylight": dict(
        mp=2.0, barrel=0.018, ca=0.0012, vignette=0.35,
        bloom_thresh=0.8, bloom_amt=0.5, bloom_sigma=6, smear=0.58,
        bayer=0.7, noise_lum=0.012, noise_chroma=0.05, chroma_nr=0.4,
        wb=(1.06, 1.0, 0.93), knee=0.85, sat=1.28, skin_magenta=0.5,
        contrast=0.22, black=0.02, sharpen=0.9, sharpen_sigma=1.0,
        jpeg_q=82, jpeg_passes=1, fmt="420",
    ),
    # The classic look: harsh on-camera flash. Hot center, dark falloff, cooler.
    "flash": dict(
        mp=2.0, barrel=0.018, ca=0.0013, vignette=0.6,
        bloom_thresh=0.7, bloom_amt=0.8, bloom_sigma=7, smear=0.72,
        bayer=0.7, noise_lum=0.02, noise_chroma=0.08, chroma_nr=0.45,
        wb=(1.02, 1.0, 1.02), knee=0.78, sat=1.18, skin_magenta=0.7,
        contrast=0.3, black=0.03, sharpen=1.0, sharpen_sigma=1.0,
        jpeg_q=78, jpeg_passes=1, fmt="420", flash=0.55,
    ),
    # Cheaper / older / higher-ISO indoor: more noise, softer, stronger artifacts.
    "lofi": dict(
        mp=1.0, barrel=0.03, ca=0.0018, vignette=0.5,
        bloom_thresh=0.78, bloom_amt=0.6, bloom_sigma=6, smear=0.95,
        bayer=0.9, noise_lum=0.035, noise_chroma=0.14, chroma_nr=0.5,
        wb=(1.05, 1.0, 0.9), knee=0.8, sat=1.15, skin_magenta=0.6,
        contrast=0.18, black=0.04, sharpen=1.2, sharpen_sigma=1.1,
        jpeg_q=68, jpeg_passes=2, fmt="420",
    ),
    # Camcorder-still grab: low res, soft, heavy chroma loss.
    "camcorder": dict(
        mp=0.35, barrel=0.025, ca=0.0016, vignette=0.45,
        bloom_thresh=0.78, bloom_amt=0.6, bloom_sigma=5, smear=0.81,
        bayer=0.6, noise_lum=0.025, noise_chroma=0.12, chroma_nr=0.7,
        wb=(1.04, 1.0, 0.95), knee=0.8, sat=1.2, skin_magenta=0.5,
        contrast=0.2, black=0.03, sharpen=0.6, sharpen_sigma=1.2,
        jpeg_q=72, jpeg_passes=1, fmt="420",
    ),

    # --- Camera profiles: modeled on documented signatures of real ~2002 digicams ---

    # Flagship "typical 2MP CCD digicam circa 2002": warm, punchy, balanced artifacts.
    "digicam": dict(
        mp=2.0, barrel=0.016, ca=0.0013, vignette=0.32,
        bloom_thresh=0.8, bloom_amt=0.5, bloom_sigma=6, smear=0.54,
        bayer=0.75, noise_lum=0.014, noise_chroma=0.05, chroma_nr=0.4,
        wb=(1.05, 1.0, 0.93), knee=0.84, sat=1.28, skin_magenta=0.5,
        contrast=0.22, black=0.02, sharpen=0.95, sharpen_sigma=1.0,
        jpeg_q=80, jpeg_passes=1, fmt="420",
    ),
    # Kodak DC / EasyShare ("Kodak Color Science"): warm, very saturated, punchy reds, soft.
    "kodak": dict(
        mp=2.1, barrel=0.018, ca=0.0012, vignette=0.35,
        bloom_thresh=0.8, bloom_amt=0.55, bloom_sigma=6, smear=0.54,
        bayer=0.8, noise_lum=0.012, noise_chroma=0.05, chroma_nr=0.45,
        wb=(1.09, 1.0, 0.89), knee=0.82, sat=1.42, skin_magenta=0.7,
        contrast=0.24, black=0.025, sharpen=0.8, sharpen_sigma=1.0,
        jpeg_q=80, jpeg_passes=1, fmt="420",
    ),
    # Sony Cyber-shot: fairly neutral / slightly cool, contrasty, heavily sharpened, clean.
    "sony": dict(
        mp=2.5, barrel=0.014, ca=0.0012, vignette=0.30,
        bloom_thresh=0.8, bloom_amt=0.45, bloom_sigma=5, smear=0.58,
        bayer=0.7, noise_lum=0.012, noise_chroma=0.045, chroma_nr=0.35,
        wb=(1.0, 1.0, 1.03), knee=0.80, sat=1.15, skin_magenta=0.3,
        contrast=0.30, black=0.03, sharpen=1.2, sharpen_sigma=1.0,
        jpeg_q=82, jpeg_passes=1, fmt="420",
    ),
    # Canon PowerShot: clean, slightly warm, balanced saturation, good detail, light artifacts.
    "canon": dict(
        mp=2.0, barrel=0.015, ca=0.0011, vignette=0.30,
        bloom_thresh=0.8, bloom_amt=0.45, bloom_sigma=6, smear=0.49,
        bayer=0.7, noise_lum=0.011, noise_chroma=0.04, chroma_nr=0.4,
        wb=(1.04, 1.0, 0.97), knee=0.83, sat=1.2, skin_magenta=0.45,
        contrast=0.22, black=0.02, sharpen=1.0, sharpen_sigma=1.0,
        jpeg_q=85, jpeg_passes=1, fmt="420",
    ),
    # Nikon Coolpix: crisp, slightly cool/green cast, a touch noisier.
    "nikon": dict(
        mp=2.0, barrel=0.016, ca=0.0014, vignette=0.32,
        bloom_thresh=0.8, bloom_amt=0.45, bloom_sigma=5, smear=0.54,
        bayer=0.75, noise_lum=0.016, noise_chroma=0.06, chroma_nr=0.4,
        wb=(0.99, 1.02, 1.0), knee=0.81, sat=1.12, skin_magenta=0.35,
        contrast=0.24, black=0.03, sharpen=1.1, sharpen_sigma=1.0,
        jpeg_q=80, jpeg_passes=1, fmt="420",
    ),
    # Fujifilm FinePix (Super CCD): vivid/velvia-like saturation, warm, smooth highlight roll-off.
    "fuji": dict(
        mp=2.5, barrel=0.015, ca=0.0012, vignette=0.30,
        bloom_thresh=0.82, bloom_amt=0.6, bloom_sigma=7, smear=0.45,
        bayer=0.7, noise_lum=0.012, noise_chroma=0.05, chroma_nr=0.4,
        wb=(1.05, 1.0, 0.95), knee=0.90, sat=1.4, skin_magenta=0.5,
        contrast=0.2, black=0.02, sharpen=0.9, sharpen_sigma=1.0,
        jpeg_q=82, jpeg_passes=1, fmt="420",
    ),
}

# AGC (automatic gain control) curves for compand: map input dB -> output dB.
# Low inputs are pulled WAY up (boosts the noise floor in quiet passages = the
# signature pumping/breathing); loud inputs are compressed (no headroom).
AGC_COMPAND = {
    "strong": "compand=attacks=0.02:decays=0.5:points=-90/-24|-50/-16|-30/-13|-15/-11|0/-9:soft-knee=6:gain=3",
    "med":    "compand=attacks=0.05:decays=0.7:points=-90/-38|-45/-22|-25/-17|0/-11:soft-knee=6:gain=2",
    "light":  "compand=attacks=0.08:decays=0.9:points=-90/-50|-40/-26|-20/-18|0/-12:soft-knee=6:gain=1",
}

# Video presets: ffmpeg geometry + codec + audio chain.
# Audio fields mirror the real capture chain of a tiny built-in electret mic:
#   ahp/alp = band-limit (no bass, rolled-off highs -> "tinny"); aagc = AGC pump;
#   ahiss = mic/circuit self-noise; abits = cheap-ADC bit depth (0 = skip);
#   arate = capture sample rate; acodec = period container codec.
VIDEO_PRESETS = {
    # ~2002 digicam movie mode: 640x480 / 15fps Motion-JPEG AVI, 11k 8-bit IMA-ADPCM mono.
    "digicam_video": dict(w=640, h=480, fps=15, codec="mjpeg", qv=8, ext=".avi",
                          interlace=False, ca=0.003, vignette=0.45, noise=12, soft=0.5, mblur=3, bloom=0.25, smear=0.5,
                          sat=1.18, contrast=1.12, warm=0.06, sharp=0.6, chroma="yuvj420p",
                          arate=11025, ahp=250, alp=6000, abits=8, ahiss=0.005,
                          aagc="strong", adrive=2.0, acodec="adpcm_ima_wav"),
    # MiniDV camcorder: 720x480 interlaced 29.97fps, 4:1:1. Better audio: 32k 16-bit PCM.
    "camcorder": dict(w=720, h=480, fps=30000 / 1001, codec="dvvideo", qv=None, ext=".avi",
                      interlace=True, ca=0.0025, vignette=0.4, noise=20, soft=0.5, mblur=1, bloom=0.30, smear=0.6,
                      dr_curve="0/0.06 0.10/0.12 0.80/0.85 1/1",   # low-light AGC: lift shadows to noisy murk (not clean black)
                      sat=1.2, contrast=1.12, warm=0.05, sharp=0.5, chroma="yuv411p",
                      arate=32000, ahp=90, alp=13000, abits=0, ahiss=0.003,
                      aagc="med", adrive=0.0, acodec="pcm_s16le", zoom_motor=0.6),
    # Low-bitrate MPEG (early SD card / web clip): heavy macroblocking + warbly low-rate MP3.
    "mpeg_lofi": dict(w=320, h=240, fps=15, codec="mpeg4", qv=None, bitrate="320k", ext=".avi",
                      interlace=False, ca=0.004, vignette=0.5, noise=16, soft=0.6, mblur=3, bloom=0.25, smear=0.5,
                      sat=1.12, contrast=1.1, warm=0.07, sharp=0.5, chroma="yuv420p",
                      arate=22050, ahp=200, alp=8000, abits=0, ahiss=0.004,
                      aagc="med", adrive=1.0, acodec="libmp3lame", abitrate="56k"),

    # --- Camera movie-mode profiles ---
    # Typical 2002 digicam movie mode (= digicam_video): MJPEG AVI + IMA-ADPCM.
    "digicam": dict(w=640, h=480, fps=15, codec="mjpeg", qv=8, ext=".avi",
                    interlace=False, ca=0.003, vignette=0.42, noise=12, soft=0.5, mblur=3, bloom=0.25, smear=0.5,
                    sat=1.22, contrast=1.12, warm=0.06, sharp=0.6, chroma="yuvj420p",
                    arate=11025, ahp=250, alp=6000, abits=8, ahiss=0.005,
                    aagc="strong", adrive=2.0, acodec="adpcm_ima_wav"),
    # Sony Cyber-shot "MPEG Movie": 320x240 @ ~16fps MPEG-1-style, tiny mono track.
    "sony": dict(w=320, h=240, fps=16, codec="mpeg4", qv=None, bitrate="550k", ext=".avi",
                 interlace=False, ca=0.003, vignette=0.4, noise=12, soft=0.5, mblur=3, bloom=0.22, smear=0.45,
                 sat=1.14, contrast=1.16, warm=0.0, sharp=0.7, chroma="yuv420p",
                 arate=16000, ahp=180, alp=7000, abits=0, ahiss=0.004,
                 aagc="med", adrive=1.0, acodec="libmp3lame", abitrate="64k"),
}


# --------------------------------------------------------------------------- #
# photo driver
# --------------------------------------------------------------------------- #
def downscale_linear(lin, out_w, out_h):
    """High-quality Lanczos downscale done in LINEAR light (correct averaging).
    Represents the sensor sampling the continuous optical image at its pixel count."""
    # Each channel must be a C-contiguous float32 buffer: PIL's "F" mode is 32-bit
    # float, so a strided or float64 slice (e.g. from an op that promoted to float64)
    # is misread as garbage/inf. ascontiguousarray with dtype=float32 guarantees both.
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(lin[..., c], dtype=np.float32),
                                        mode="F").resize((out_w, out_h), Image.LANCZOS))
             for c in range(lin.shape[2])]
    return np.stack(chans, -1).astype(np.float32)


def process_photo(in_path, out_path, p, datestamp=None, strength=1.0, seed=12345):
    img = Image.open(in_path).convert("RGB")
    W0, H0 = img.size

    x = np.asarray(img, np.float32) / 255.0
    lin = srgb_to_linear(x)
    rng = np.random.default_rng(seed)

    def amt(v):  # global strength scaler for "amount"-like params
        return v * strength

    # The pipeline follows the real imaging chain, in order:
    #   scene light -> optics -> sensor sampling -> sensor capture -> ISP -> JPEG
    # Optical effects act on the full-resolution continuous image BEFORE the sensor
    # samples it (downscale), which is why CA/distortion/vignette come first.

    # --- (a) scene illumination: flash lights the scene before it reaches the lens ---
    if p.get("flash"):
        H, W = lin.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - (W - 1) / 2) / ((W - 1) / 2)) ** 2 +
                    ((yy - (H - 1) / 2) / ((H - 1) / 2)) ** 2) / np.sqrt(2)
        lin = lin * (1 + amt(p["flash"]) * (1 - r) ** 2)[..., None]

    # --- (b) optics (continuous image): distortion + lateral CA, then vignetting ---
    lin = lens_distort_ca(lin, p["barrel"], amt(p["ca"]))
    lin = vignette(lin, amt(p["vignette"]))

    # --- (c) sensor sampling: the CCD captures the optical image at its pixel count ---
    if p["mp"]:
        target = p["mp"] * 1e6
        H, W = lin.shape[:2]
        if W * H > target:
            s = (target / (W * H)) ** 0.5
            lin = downscale_linear(lin, max(1, int(W * s)), max(1, int(H * s)))

    # --- (d) sensor capture: highlight bloom/smear, full-well clip, photosite noise,
    #         then CFA mosaic + demosaic (so the noise is correlated by demosaicing) ---
    L, clip = light_source_map(lin)
    lin = highlight_bloom(lin, L, amt(p["bloom_amt"]), p["bloom_sigma"])
    lin = ccd_smear(lin, clip, amt(p["smear"]))
    lin = highlight_knee(lin, p["knee"])                                  # full-well roll-off (pre-WB)
    lin = add_noise(lin, amt(p["noise_lum"]), amt(p["noise_chroma"]), rng)
    lin = bayer_emulate(lin, p["bayer"] * strength)

    # --- (e) ISP: white balance, then display-referred color/tone/NR/sharpen ---
    lin = lin * np.array(p["wb"], np.float32)
    x = linear_to_srgb(lin)
    x = saturate(x, 1 + (p["sat"] - 1) * strength, p["skin_magenta"])
    x = s_curve(x, p["contrast"] * strength, p["black"])
    x = chroma_nr(x, amt(p["chroma_nr"]))
    x = unsharp(x, amt(p["sharpen"]), p["sharpen_sigma"])

    out = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")

    if datestamp:
        draw_datestamp(out, datestamp)

    # JPEG encode: 4:2:0 subsampling + quantization (the final, signature artifact).
    sub = {"420": 2, "422": 1, "444": 0}[p["fmt"]]
    out.save(out_path, "JPEG", quality=p["jpeg_q"], subsampling=sub)
    for _ in range(p.get("jpeg_passes", 1) - 1):  # generational re-save = more blocking
        Image.open(out_path).save(out_path, "JPEG", quality=p["jpeg_q"], subsampling=sub)
    return out.size


def draw_datestamp(img, text):
    """Orange seven-segment-style date stamp, bottom-right, with a slight glow."""
    draw = ImageDraw.Draw(img)
    W, H = img.size
    size = max(11, int(H * 0.030))
    font = None
    for cand in ("DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf"):
        try:
            font = ImageFont.truetype(cand, size); break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = W - tw - int(W * 0.03); y = H - th - int(H * 0.05)
    glow = (255, 140, 0)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=(120, 50, 0))
    draw.text((x, y), text, font=font, fill=glow)


# --------------------------------------------------------------------------- #
# video driver (ffmpeg)
# --------------------------------------------------------------------------- #
def video_ca_radial(in_lbl, out_lbl, e, w, h):
    """Radial lateral CA for video: magnify the red plane and shrink the blue plane
    about the center, so the color split is zero at center and grows toward the edges,
    like a real lens. ffmpeg's rgbashift only does a uniform whole-frame shift,
    which looks like an anaglyph; this avoids that. `e` ~ 0.003 gives a couple px at
    the edge of a 640px frame.
    """
    rs, ss = 1.0 + e, 1.0 - e
    # setsar=1 on every branch: scale rewrites the sample aspect ratio, and mergeplanes
    # refuses to combine planes whose SAR disagrees.
    return (f"[{in_lbl}]format=rgb24,extractplanes=r+g+b[car][cag][cab];"
            f"[car]scale=w=iw*{rs:.5f}:h=ih*{rs:.5f},crop={w}:{h},setsar=1[cars];"   # red drifts out
            f"[cag]setsar=1[cags];"
            f"[cab]scale=w=iw*{ss:.5f}:h=ih*{ss:.5f},"
            f"pad={w}:{h}:({w}-iw)/2:({h}-ih)/2,setsar=1[cabs];"                     # blue drifts in
            f"[cars][cags][cabs]mergeplanes=0x102000:gbrp[{out_lbl}]")


def video_smear(in_lbl, out_lbl, amount, h):
    """CCD vertical smear for video: a saturated source leaks charge down its column.

    Detection uses BRIGHTNESS *and* CONTRAST: 'prominence' = luma - (wide local average),
    so a bright object on a dark background smears even when it never reaches near-white,
    while a flat wall does not.

    Colour is physically modelled like the photo path: the streak is ADDED (it is
    overflow charge), and its tint depends on charge. The green channel lags at low
    charge (so a faint streak reads magenta/purple), then catches up as charge rises, so
    a bright streak floods all channels and washes out to WHITE rather than staying a
    saturated purple."""
    sy = max(16, h // 4)                                  # vertical reach (box average dilutes, so keep moderate)
    bg = max(40, h // 3)                                  # wide local average so isolated bright objects stand out
    return (
        f"[{in_lbl}]split=3[sm0][smA][smB];"
        f"[smB]format=gray,gblur=sigma={bg}[smavg];"      # local average brightness
        f"[smA]format=gray[smg];"
        f"[smg][smavg]blend=all_mode=subtract[smc];"      # prominence = luma - local avg (charge)
        f"[smc]curves=all='0/0 0.04/0 0.16/1 1/1',"      # floor out noise, saturate real sources to full charge
        f"avgblur=sizeX=1:sizeY={sy}[q];"                 # spread down/up the column, stay thin
        f"[q]split[qa][qb];"
        f"[qb]curves=all='0/0 0.5/0.16 0.8/0.5 1/1'[qg];"  # green lags -> purple when faint, white when bright
        f"[qg][qa]mergeplanes=0x001010:gbrp,"            # R,B = charge; G = lagging green
        f"colorchannelmixer=rr={amount*3:.3f}:gg={amount*3:.3f}:bb={amount*3:.3f}[smt];"  # gain (box-blur dilutes)
        f"[sm0][smt]blend=all_mode=addition[{out_lbl}]")  # additive -> bright cores clip to white


def build_post_filters(vp, datestamp):
    """Filters AFTER bloom/smear, in imaging order:
        sensor (limited DR -> read noise) -> ISP (white balance -> sat/contrast ->
        sharpen) -> scan/overlay (interlace, datestamp).
    Vignetting is optical and is done back in the geometry stage, not here."""
    h = vp["h"]
    filters = [
        # --- sensor response ---
        # limited CCD dynamic range. Default crushes shadows + rolls highlights to kill
        # the flat modern look. Camcorders in low light instead *lift* the shadows
        # (AGC gain-up) into noisy murk -> presets can override `dr_curve`.
        f"curves=master='{vp.get('dr_curve', '0/0 0.12/0.05 0.78/0.86 1/1')}'",
        f"noise=alls={vp['noise']}:allf=t+u",                           # read noise (before ISP sharpen)
        # --- ISP ---
        f"colorbalance=rm={vp['warm']}:bm=-{vp['warm']}",                # white balance first
        f"eq=saturation={vp['sat']}:contrast={vp['contrast']}",         # then color matrix / tone
        f"unsharp=5:5:{vp['sharp']}:5:5:0.0",                            # sharpening (amplifies the noise above)
    ]
    if vp.get("interlace"):
        filters.append("interlace=scan=tff:lowpass=1")                  # combing on motion
    if datestamp:
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
        fontopt = f":fontfile={font}" if os.path.exists(font) else ""
        filters.append(
            f"drawtext=text='{datestamp}'{fontopt}:fontcolor=orange:fontsize={max(14,int(h*0.05))}"
            f":x=w-tw-12:y=h-th-12:shadowcolor=0x802000:shadowx=1:shadowy=1")
    return ",".join(filters)


def build_video_graph(vp, datestamp):
    """Full video chain '[0:v]...[v]', in true imaging order:

        sampling/exposure -> OPTICS -> SENSOR -> ISP -> scan/encode
        crop4:3 + scale + motion-blur + fps        (sampling & exposure)
        -> soft + vignette + radial CA             (optics)
        -> bloom + smear + limited-DR + noise       (sensor)
        -> white balance + sat/contrast + sharpen   (ISP)
        -> interlace + datestamp                    (scan/overlay)

    The relative order matters: optics before sensor before ISP; noise is added
    before sharpening (so sharpening amplifies it); white balance precedes the
    color/tone; vignette is optical so it precedes the sensor highlight effects.
    Frame-rate drop is done up front so the heavy stages run on fewer frames; for
    interlaced presets we feed 2x then interlace weaves two frames -> one.

    Authenticity notes (these stop it reading as 'filtered HD'):
      * crop to 4:3 (a 4:3 sensor framed a narrower FOV; it did NOT squish 16:9),
      * motion blur (low-fps capture integrates light; frames aren't razor-sharp),
      * optical softness (cheap small lens), bloom + vertical smear (CCD highlights).
    """
    w, h = vp["w"], vp["h"]
    interlace = vp.get("interlace")
    fps = vp["fps"] * 2 if interlace else vp["fps"]

    # --- sampling + exposure, then OPTICS (softness, vignetting) ---
    # 4:3 sensor FOV (no stretch): largest centered 4:3 box that fits any input
    # aspect, so portrait or already-4:3 sources work too (commas escaped for ffmpeg).
    g1 = [r"crop=w=min(iw\,ih*4/3):h=min(ih\,iw*3/4)",
          f"scale={w}:{h}:flags=lanczos",
          "setdar=4/3"]                          # display 4:3 even for 720x480 non-square pixels
    if not interlace and vp.get("mblur", 1) > 1:
        g1.append(f"tmix=frames={vp['mblur']}")  # low-fps capture motion blur (exposure)
    g1.append(f"fps={fps}")
    if vp.get("soft", 0) > 0:
        g1.append(f"gblur=sigma={vp['soft']}")   # optics: cheap-lens softness
    # optics: light falloff. ffmpeg's vignette angle grows -> darker corners (default
    # PI/5 ≈ 0.628). Map the 0..1 amount to a GENTLE 0.40..0.58 rad (below default) so
    # it's a subtle digicam vignette, not the heavy PI/3 an int()-truncated formula gave.
    g1.append(f"vignette=a={0.30 + 0.55 * vp['vignette']:.3f}")
    stmts = ["[0:v]" + ",".join(g1) + "[geo]"]
    cur = "geo"

    # --- optics: radial chromatic aberration ---
    if vp.get("ca", 0) > 0:
        stmts.append(video_ca_radial(cur, "vca", vp["ca"], w, h)); cur = "vca"

    # --- sensor: highlight bloom (screen-blend a blurred copy) ---
    if vp.get("bloom", 0) > 0:
        stmts.append(f"[{cur}]split[bb0][bb1];[bb1]gblur=sigma=7[bb2];"
                     f"[bb0][bb2]blend=all_mode=screen:all_opacity={vp['bloom']}[blm]")
        cur = "blm"

    # --- sensor: CCD vertical purple smear (charge leak from clipped highlights) ---
    if vp.get("smear", 0) > 0:
        stmts.append(video_smear(cur, "smr", vp["smear"], h)); cur = "smr"

    # --- sensor response + ISP + scan/overlay ---
    stmts.append(f"[{cur}]{build_post_filters(vp, datestamp)}[v]")
    return ";".join(stmts)


def build_audio_graph(vp, motor=False):
    """Audio chain in true capture order. Returns a filter_complex fragment ending in [a].

    Order matters: the mic's self-noise is analog, added at the mic/preamp BEFORE the
    AGC. Hiss is mixed in before compand, so the AGC pumps the noise floor up in quiet
    passages (the breathing artifact). The ADC's
    bit-crush + sample-rate come AFTER the AGC, because quantization happens at the
    converter, downstream of the analog gain stage. If `motor`, a zoom-motor track on
    input 1 is mixed in here too (it is mechanical noise the mic picks up, so it sits
    before the band-limit and AGC like everything else).
        mono (+ motor) -> +mic hiss -> band-limit -> AGC -> preamp clip -> ADC(bits+rate)
    """
    mono = "[0:a]aformat=channel_layouts=mono[m]"                         # single built-in mic
    hiss = (f"anoisesrc=color=pink:amplitude={vp['ahiss']}:sample_rate=48000,"
            "aformat=channel_layouts=mono[hs]")                          # mic/circuit self-noise
    pre = [mono, hiss]
    if motor:
        pre.append("[1:a]aformat=channel_layouts=mono[mo]")              # zoom-motor whir (input 1)
        mix = "[m][mo][hs]amix=inputs=3:duration=first:dropout_transition=0:normalize=0[mix]"
    else:
        mix = "[m][hs]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]"
    chain = [f"highpass=f={vp['ahp']}", f"lowpass=f={vp['alp']}",         # mic band (no bass/highs)
             AGC_COMPAND[vp["aagc"]]]                                     # AGC pumps signal+hiss
    if vp.get("adrive", 0) > 0:
        chain += [f"volume={vp['adrive']}dB", "alimiter=limit=0.97:level=disabled"]  # preamp clip
    if vp.get("abits"):
        chain.append(f"acrusher=bits={vp['abits']}:samples=1:mode=log:mix=1")        # ADC bit depth
    chain.append(f"aresample={vp['arate']}")                              # ADC sample rate
    final = "[mix]" + ",".join(chain) + "[a]"
    return ";".join(pre + [mix, final])


def media_duration(path):
    """Duration in seconds via ffprobe, or 0.0 if unknown."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nk=1:nw=1", path], capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def run_ffmpeg(cmd, label, duration=0.0):
    """Run ffmpeg and show a rich progress bar parsed from its -progress output.
    `cmd` should already include: -hide_banner -loglevel error -progress pipe:1 -nostats."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    cols = (TextColumn("[bold cyan]{task.description}"), BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"), TimeRemainingColumn())
    with Progress(*cols, console=console, transient=True) as prog:
        task = prog.add_task(label, total=duration or None)
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time=") and duration:
                t = line.split("=", 1)[1]
                try:
                    h, m, s = t.split(":")
                    prog.update(task, completed=min(duration, int(h) * 3600 + int(m) * 60 + float(s)))
                except Exception:
                    pass
            elif line == "progress=end":
                prog.update(task, completed=duration or 1)
    err = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        console.print("[red]ffmpeg failed:[/red]")
        console.print(err.strip())
        sys.exit(1)


def process_video(in_path, out_path, vp, datestamp=None, audio=True, motor_wav=None):
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (needed for video).")
    do_audio = audio and has_audio(in_path)
    use_motor = bool(motor_wav) and do_audio

    graph = build_video_graph(vp, datestamp)
    if do_audio:
        graph += ";" + build_audio_graph(vp, motor=use_motor)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", in_path]
    if use_motor:
        cmd += ["-i", motor_wav]                 # input 1: the zoom-motor track
    cmd += ["-filter_complex", graph, "-map", "[v]",
           "-aspect", "4:3",                       # force 4:3 display (DV/MPEG store the flag)
           "-pix_fmt", vp["chroma"], "-c:v", vp["codec"]]
    if vp["codec"] == "mjpeg":
        cmd += ["-q:v", str(vp["qv"]), "-huffman", "optimal"]
    elif vp["codec"] == "mpeg4":
        cmd += (["-b:v", vp["bitrate"]] if vp.get("bitrate") else ["-q:v", "6"]) + ["-mbd", "rd"]
    if vp.get("interlace"):
        cmd += ["-flags", "+ilme+ildct"]
    if do_audio:
        cmd += ["-map", "[a]", "-c:a", vp["acodec"], "-ar", str(vp["arate"]), "-ac", "1"]
        if vp["acodec"] == "libmp3lame":
            cmd += ["-b:a", vp.get("abitrate", "64k")]
    else:
        cmd += ["-an"]
    cmd += ["-progress", "pipe:1", "-nostats", out_path]
    run_ffmpeg(cmd, "developing video", media_duration(in_path))


def audio_ext(vp):
    """File extension for an audio-only output, matching the preset's codec."""
    return ".mp3" if vp["acodec"] == "libmp3lame" else ".wav"


def process_audio(in_path, out_path, vp):
    """Audio-only: run a source sound through the built-in-mic degradation chain
    (mono -> hiss -> band-limit -> AGC -> ADC bit-crush/sample-rate) and encode with
    the preset's period codec. Same chain used for a video's soundtrack."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (needed for audio).")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", in_path,
           "-filter_complex", build_audio_graph(vp),
           "-map", "[a]", "-c:a", vp["acodec"], "-ar", str(vp["arate"]), "-ac", "1"]
    if vp["acodec"] == "libmp3lame":
        cmd += ["-b:a", vp.get("abitrate", "64k")]
    cmd += ["-progress", "pipe:1", "-nostats", out_path]
    run_ffmpeg(cmd, "developing audio", media_duration(in_path))


def has_audio(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "json", path],
            capture_output=True, text=True)
        return bool(json.loads(out.stdout or "{}").get("streams"))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# zoom detection + motor sound
# --------------------------------------------------------------------------- #
# Old digicams/camcorders zoomed with a small electric servo right next to the
# built-in mic, so the mic mechanically picked up its whir/buzz (and the AGC made
# it worse). We detect zoom from the video (the frame scaling about its center over
# time) and synthesize a matching motor whir into the mic chain.

def _scale_about_center(img, c):
    """Scale a 2D uint8/float frame by factor c about its center, same output size."""
    H, W = img.shape
    nw, nh = max(1, round(W * c)), max(1, round(H * c))
    im = Image.fromarray(img.astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (W, H))
    canvas.paste(im, ((W - nw) // 2, (H - nh) // 2))   # PIL crops/pads as needed
    return np.asarray(canvas, np.float32)


def _translation(a, b):
    """Integer (dx, dy) that best shifts `a` onto `b`, via phase correlation. Phase
    correlation keys on the cross-power spectrum, which depends only on translation, so
    this is robust and lets us separate handheld panning (pure translation) from zoom
    (scale about the center)."""
    A = np.fft.rfft2(a); B = np.fft.rfft2(b)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-6
    c = np.fft.irfft2(R, s=a.shape)
    dy, dx = np.unravel_index(int(np.argmax(c)), c.shape)
    H, W = a.shape
    if dy > H // 2: dy -= H
    if dx > W // 2: dx -= W
    return int(dx), int(dy)


def detect_zoom(in_path, analyze_fps=6):
    """Return a list of (start_s, end_s, speed) zoom segments.

    For each frame pair we find the scale-about-center that best matches the next frame.
    Two guards keep handheld motion, pans and shake from registering as zoom:
      * the best scale must fit clearly better than no-scaling (ratio test) -- a pan or
        translation is not improved by scaling, so it is rejected; and
      * a kept segment must have a real, sustained directional scale change (the net
        zoom over the segment exceeds a few percent) -- jitter alternates sign and
        cancels out.
    """
    W, H = 128, 72
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", in_path, "-vf",
         f"fps={analyze_fps},scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    n = len(raw) // (W * H)
    if n < 5:
        return []
    frames = np.frombuffer(raw[:n * W * H], np.uint8).reshape(n, H, W).astype(np.float32)
    cand = np.linspace(0.94, 1.06, 7)                        # inter-frame scale guesses (1.0 = middle)
    mid = len(cand) // 2
    y0, y1, x0, x1 = int(H * 0.25), int(H * 0.75), int(W * 0.25), int(W * 0.75)  # center window
    best = np.ones(n - 1, np.float32)
    ratio = np.ones(n - 1, np.float32)                       # err(best scale) / err(no scale)
    for i in range(n - 1):
        # remove handheld translation first: align frame i onto i+1 by the estimated
        # shift, so the leftover only-improves-with-scaling part is the actual zoom.
        dx, dy = _translation(frames[i], frames[i + 1])
        a = np.roll(np.roll(frames[i], dy, axis=0), dx, axis=1)
        b = frames[i + 1][y0:y1, x0:x1]
        err = [np.mean((_scale_about_center(a, c)[y0:y1, x0:x1] - b) ** 2) for c in cand]
        j = int(np.argmin(err))
        best[i] = cand[j]
        ratio[i] = err[j] / (err[mid] + 1e-6)
    if len(best) >= 3:
        best = np.convolve(best, np.ones(3) / 3, mode="same")
    rate = np.log(np.clip(best, 1e-3, None))                 # >0 zoom in, <0 zoom out
    zooming = (ratio < 0.78) & (np.abs(rate) > np.log(1.01))   # scaling must genuinely help
    dt = 1.0 / analyze_fps
    segs, run_start, gap = [], None, 0
    for i in range(len(rate)):
        if zooming[i]:
            run_start = i if run_start is None else run_start
            gap = 0
        elif run_start is not None:
            gap += 1
            if gap > 2:
                segs.append((run_start, i - gap)); run_start = None
    if run_start is not None:
        segs.append((run_start, len(rate) - 1))
    out = []
    for a, b in segs:
        if (b - a + 1) * dt < 0.6:                           # too short to be a deliberate zoom
            continue
        net = float(np.sum(rate[a:b + 1]))                   # net directional scale change
        if abs(net) < np.log(1.12):                          # < ~12% total -> handheld drift, not a zoom
            continue
        speed = abs(net) / ((b - a + 1) * dt)                # log-scale per second
        out.append((a * dt, (b + 1) * dt, speed))
    return out


def synth_motor_track(total_dur, intervals, sr=48000, intensity=0.5, seed=7):
    """Synthesize a mono motor-whir track: a buzzy harmonic tone (servo + gear mesh)
    plus mechanical noise, enveloped over each zoom segment. The mic chain band-limits
    and AGC-pumps it downstream, like a real built-in mic would."""
    rng = np.random.default_rng(seed)
    track = np.zeros(int(total_dur * sr) + sr, np.float32)
    for t0, t1, speed in intervals:
        n = int((t1 - t0) * sr)
        if n <= 0:
            continue
        t = np.arange(n) / sr
        f0 = 130.0 + 45.0 * float(np.clip(speed, 0, 3))      # faster zoom -> slightly higher pitch
        vib = 1.0 + 0.015 * np.sin(2 * np.pi * 7 * t)        # motor not perfectly steady
        buzz = sum(np.sin(2 * np.pi * f0 * k * vib * t) / k for k in range(1, 7)) / 2.0
        noise = rng.standard_normal(n).astype(np.float32) * 0.5
        sig = (0.7 * buzz + 0.3 * noise).astype(np.float32)
        env = np.ones(n, np.float32)                         # attack / release
        a = min(n, int(0.06 * sr)); r = min(n, int(0.10 * sr))
        env[:a] = np.linspace(0, 1, a); env[-r:] = np.linspace(1, 0, r)
        sig *= env * intensity
        s = int(t0 * sr)
        m = min(n, len(track) - s)
        track[s:s + m] += sig[:m]
    return np.clip(track, -1, 1)


def write_wav(path, arr, sr=48000):
    pcm = (np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)


# --------------------------------------------------------------------------- #
# cli (Typer)
# --------------------------------------------------------------------------- #
import typer

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXT = {".mov", ".mp4", ".mkv", ".avi", ".webm", ".m4v", ".y4m", ".mts"}
AUD_EXT = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma"}

PRESET_DESC = {  # photo
    "digicam": "typical 2MP CCD digicam, warm, punchy (default)",
    "kodak": "Kodak Color Science, warm, very saturated",
    "sony": "Cyber-shot, neutral-cool, contrasty, sharp",
    "canon": "PowerShot, clean, slightly warm, balanced",
    "nikon": "Coolpix, crisp, slightly cool/green",
    "fuji": "FinePix Super CCD, vivid, smooth highlights",
    "daylight": "generic outdoor CCD, punchy",
    "flash": "harsh on-camera flash, hot center, dark falloff",
    "lofi": "cheap / high-ISO indoor, noisy, soft",
    "camcorder": "low-res soft video-still grab",
}
VIDEO_DESC = {
    "digicam": "digicam movie mode, MJPEG 640x480 + IMA-ADPCM",
    "digicam_video": "same as digicam",
    "sony": "Cyber-shot MPEG Movie, 320x240 @16fps",
    "camcorder": "MiniDV, interlaced 720x480, low-light grain",
    "mpeg_lofi": "low-bitrate MPEG-4 320x240, heavy macroblocking",
}

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help="Give photos, video and audio an authentic early-2000s digital-camera look.")


def _preset_table(title, names, desc):
    table = Table(title=title, title_justify="left", title_style="bold",
                  header_style="bold cyan", border_style="dim", pad_edge=False)
    table.add_column("preset", style="green", no_wrap=True)
    table.add_column("look")
    for k in names:
        table.add_row(k, desc.get(k, ""))
    return table


def _print_presets():
    console.print()
    console.print(_preset_table("Photo presets", PRESETS, PRESET_DESC))
    console.print()
    console.print(_preset_table("Video / audio presets", VIDEO_PRESETS, VIDEO_DESC))
    console.print()


def _version(value: bool):
    if value:
        typer.echo(f"digicam2000 {__version__}")
        raise typer.Exit()


@app.command()
def convert(
    input: Optional[Path] = typer.Argument(
        None, exists=True, dir_okay=False, help="Source image, video, or audio file."),
    output: Optional[Path] = typer.Argument(
        None, help="Output path. Default: <input>.digicam.<ext>"),
    preset: str = typer.Option("digicam", "--preset", "-p", help="Look preset (see [b]--list[/b])."),
    mp: Optional[float] = typer.Option(None, "--mp", help="Target megapixels (photo)."),
    barrel: Optional[float] = typer.Option(None, "--barrel", help="Barrel distortion k (photo); ~0.02 subtle, 0 = off."),
    strength: float = typer.Option(1.0, "--strength", "-s", min=0.0, max=1.5, help="Global intensity 0..1.5 (photo)."),
    datestamp: Optional[str] = typer.Option(None, "--datestamp", "-d", help="Orange corner date, e.g. 2002-07-04."),
    no_audio: bool = typer.Option(False, "--no-audio", help="Strip audio instead of degrading it (video)."),
    zoom_sound: bool = typer.Option(True, "--zoom-sound/--no-zoom-sound",
                                    help="Add zoom-motor whir when a zoom is detected (video)."),
    seed: int = typer.Option(12345, "--seed", help="Noise RNG seed (photo)."),
    list_presets: bool = typer.Option(False, "--list", "-l", help="List all presets and exit."),
    _v: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version and exit."),
):
    """Convert an image, video or audio file to a 2000s digicam look (auto-detected by extension)."""
    if list_presets:
        _print_presets()
        raise typer.Exit()
    if input is None:
        typer.secho("Error: missing INPUT file (or pass --list to see presets).", fg="red", err=True)
        raise typer.Exit(2)

    ext = input.suffix.lower()
    if ext in VID_EXT:
        if preset not in VIDEO_PRESETS:
            typer.secho(f"note: '{preset}' has no video profile; using 'digicam_video'.", fg="yellow")
        vp = dict(VIDEO_PRESETS.get(preset, VIDEO_PRESETS["digicam_video"]))
        out = str(output) if output else str(input.with_suffix("")) + ".digicam" + vp["ext"]
        motor_wav = None
        if zoom_sound and not no_audio and has_audio(str(input)):
            with console.status("[cyan]scanning for zoom[/]", spinner="dots"):
                segs = detect_zoom(str(input))
            if segs:
                total = sum(b - a for a, b, _ in segs)
                console.print(f"[cyan]zoom detected[/] {len(segs)} segment(s), {total:.1f}s "
                              f"-> adding motor sound")
                motor_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                write_wav(motor_wav, synth_motor_track(media_duration(str(input)), segs,
                                                       intensity=vp.get("zoom_motor", 0.35)))
        try:
            process_video(str(input), out, vp, datestamp, audio=not no_audio, motor_wav=motor_wav)
        finally:
            if motor_wav:
                os.unlink(motor_wav)
    elif ext in AUD_EXT:
        if preset not in VIDEO_PRESETS:
            typer.secho(f"note: '{preset}' has no audio profile; using 'digicam_video'.", fg="yellow")
        vp = dict(VIDEO_PRESETS.get(preset, VIDEO_PRESETS["digicam_video"]))
        out = str(output) if output else str(input.with_suffix("")) + ".digicam" + audio_ext(vp)
        process_audio(str(input), out, vp)
    elif ext in IMG_EXT:
        if preset not in PRESETS:
            typer.secho(f"Error: unknown photo preset '{preset}'. Try --list.", fg="red", err=True)
            raise typer.Exit(2)
        p = dict(PRESETS[preset])
        if mp is not None:
            p["mp"] = mp
        if barrel is not None:
            p["barrel"] = barrel
        out = str(output) if output else str(input.with_suffix("")) + ".digicam.jpg"
        with console.status("[bold cyan]developing photo[/]", spinner="dots"):
            w, h = process_photo(str(input), out, p, datestamp, strength, seed)
        console.print(f"[green]wrote[/] {out}  [dim]{w}x{h}[/]")
        return
    else:
        typer.secho(f"Error: unsupported file type '{ext}'.", fg="red", err=True)
        raise typer.Exit(2)
    console.print(f"[green]wrote[/] {out}")


if __name__ == "__main__":
    app()
