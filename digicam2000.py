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
import os, sys, subprocess, shutil, json, wave, tempfile, random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

__version__ = "1.0.0"
console = Console()

# A camera's defect map is a property of the silicon, so the hot/stuck pixels and the
# readout fixed-pattern noise are keyed to a FIXED seed: they land in the same place on
# every render, like the same camera being used again. Everything else (shot/read grain,
# chroma blotches, tape head-switch hash, mic hiss) is fresh thermal noise and gets a new
# random seed each render, unless the user pins one with --seed.
SENSOR_SEED = 12345


def _resolve_seed(seed):
    """A concrete seed plus a private RNG for deriving per-stage sub-seeds. If `seed` is
    None we draw a fresh one so each render differs; if it is given, the render is
    reproducible (the sensor defects are fixed regardless)."""
    if seed is None:
        seed = random.randrange(1 << 32)
    return seed, random.Random(seed)

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


def ccd_smear(lin, clip, amount, mode="classic"):
    """Vertical charge-leak streaks from *saturated* pixels (the CCD tell), in linear light.

    Two models:
      * "classic" (default): the original, a uniform magenta/purple streak added along the
        column. Simple and reads well.
      * "physical" (WIP): the streak is overflow CHARGE, so its colour depends on amount,
        white at a bright core and grading to purple in the dim tail. Still being tuned.
    """
    if amount <= 0:
        return lin
    over = np.clip(clip, 0, 1) ** 1.2
    decay = 0.985 if mode == "physical" else 0.97
    up = np.empty_like(over); acc = np.zeros_like(over[0])
    for i in range(over.shape[0]):
        acc = np.maximum(over[i], acc * decay); up[i] = acc
    dn = np.empty_like(over); acc = np.zeros_like(over[0])
    for i in range(over.shape[0] - 1, -1, -1):
        acc = np.maximum(over[i], acc * decay); dn[i] = acc
    streak = np.maximum(up, dn)
    if mode == "physical":
        q = streak * amount * 4.0                 # scale up for the additive charge model
        t = np.clip(q / 0.40, 0, 1)[..., None]    # 0 -> purple, 1 -> white
        purple = np.array([0.85, 0.40, 1.0], np.float32)
        return lin + q[..., None] * (purple * (1 - t) + t)
    tint = np.array([0.55, 0.35, 0.95], np.float32)   # classic uniform magenta/purple
    return lin + (streak * amount)[..., None] * tint


def purple_fringe(lin, clip, amount):
    """Brightness-dependent chromatic fringing: the purple/magenta halo a cheap lens
    throws onto a high-contrast edge that borders a blown-out highlight.

    This is a different mechanism from the lateral CA in lens_distort_ca. Lateral CA is
    geometric -- a radial, brightness-independent colour split that grows with field
    height. Purple fringing is axial (longitudinal) CA plus a little sensor blooming, so
    it appears ONLY where a dark edge meets an overexposed area and scales with the local
    contrast and the highlight's brightness. We key it to the saturation/clip map: how
    close a pixel is to a blown highlight, times the local edge gradient, tinted violet.
    That makes it strictly brightness/contrast-dependent, as the physics requires."""
    if amount <= 0:
        return lin
    near = gauss_blur(np.clip(clip, 0, 1)[..., None], 3.0)[..., 0]   # proximity to a blown highlight
    lum = lin.mean(2)
    gx = np.abs(np.gradient(lum, axis=1))
    gy = np.abs(np.gradient(lum, axis=0))
    edge = np.clip((gx + gy) * 4.0, 0, 1)                            # local contrast
    fringe = (near * edge)[..., None]
    tint = np.array([0.6, 0.0, 1.0], np.float32)                     # violet/magenta
    return np.clip(lin + fringe * amount * 0.15 * tint, 0, None)


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


def hot_pixels(lin, rng, density, amount):
    """Stuck / hot photosites: fixed defective pixels that leak charge.

    A real CCD has a handful of sensels whose dark current is far above their
    neighbours'. On a bright daylight frame they hide in the signal, but on a
    long or high-ISO exposure they cross threshold and show as fixed specks:
    single-channel R/G/B "stuck" pixels, or near-white "hot" ones. Their
    positions come from the seed, so they sit in the same place every frame,
    exactly like a given camera's defect map. Most visible in the shadows.
    """
    if density <= 0 or amount <= 0:
        return lin
    H, W = lin.shape[:2]
    n = max(1, int(round(density * (H * W) / 1e6)))   # defects per megapixel
    ys = rng.integers(0, H, n); xs = rng.integers(0, W, n)
    mag = (amount * (0.5 + rng.random(n))).astype(np.float32)  # charge over threshold
    ch = rng.integers(0, 3, n)
    stuck = rng.random(n) < 0.6                        # 60% single-channel, 40% white-hot
    out = lin.copy()
    np.add.at(out, (ys[stuck], xs[stuck], ch[stuck]), mag[stuck])
    hot = ~stuck
    for c in range(3):
        np.add.at(out, (ys[hot], xs[hot], c), mag[hot])
    return np.clip(out, 0, None)


def fixed_pattern_noise(lin, rng, amount):
    """CCD readout fixed-pattern noise: a faint, frame-constant stripe pattern.

    Charge is clocked out through a serial register into one or two shared
    output amplifiers, so each column carries a small constant gain/offset from
    its readout path; dark-current non-uniformity adds a weaker per-row term.
    The result is a stationary vertical-stripe texture that's invisible in
    bright detail but shows in flat shadows and skies. Fixed by the seed.
    """
    if amount <= 0:
        return lin
    H, W = lin.shape[:2]
    col_gain = 1.0 + rng.standard_normal(W).astype(np.float32) * amount * 0.5
    col_off = rng.standard_normal(W).astype(np.float32) * amount * 0.01
    row_off = rng.standard_normal(H).astype(np.float32) * amount * 0.004
    out = lin * col_gain[None, :, None] + col_off[None, :, None] + row_off[:, None, None]
    return np.clip(out, 0, None)


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


def tone_banding(x, levels):
    """Posterize to a reduced number of tone steps -> 8-bit-JPEG banding.

    Early digicams captured straight to 8-bit JPEG, so a smooth gradient only
    had ~256 levels to work with and broke into visible bands in skies and
    shadows. Quantizing to fewer levels reproduces it: textured areas are
    dithered by the sensor noise added earlier and stay smooth, while the flat
    gradients step, just like the period artifact.
    """
    if not levels or levels >= 256:
        return x
    return np.clip(np.round(x * (levels - 1)) / (levels - 1), 0, 1)


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #
def iso_sensor_defaults(iso):
    """Physically-motivated static-defect levels (hot pixels, FPN, purple-fringe) for a
    given ISO/gain, so EVERY preset carries the sensor tells, scaled by how the camera
    was pushed -- not just the night preset.

    The unifying factor is analog gain. A high ISO applies more gain to the same dark
    frame, so the additive dark-current family is amplified and more of it crosses the
    visible threshold:
      * hot/stuck pixels -- anomalous dark current; their count and brightness rise with
        gain and integration time (dark current roughly doubles every 6-8 C and is
        multiplied by ISO). Additive, so they read out of the shadows, not the highlights.
      * FPN -- its dark-offset term (DSNU) scales with the same gain; the multiplicative
        PRNU term is ~constant, so total FPN grows sub-linearly (sqrt-ish) with gain.
      * purple fringe -- a cheap-lens/contrast effect, not a gain effect, so it is derived
        from the lens CA elsewhere, not here.
    Anchored so ISO 100 is a clean daylight frame (a couple of near-invisible defects) and
    ISO 800 is a murky high-gain/long exposure. (Refs: photon-transfer + dark-current and
    PRNU/DSNU literature.)"""
    g = iso / 100.0
    hot_px = round(4.0 * g, 1)                                # defects/megapixel, ~prop. to gain
    hot_amt = round(min(0.95, 0.55 + 0.13 * np.log2(max(1.0, g))), 2)
    fpn = round(0.02 * (g ** 0.5), 3)                         # DSNU-dominated -> sqrt with gain
    return hot_px, hot_amt, fpn


def _fill_sensor_defects(presets, kind):
    """Give every preset the ISO-scaled static defects, without clobbering any a preset
    set explicitly (e.g. the night presets keep their hand-tuned values). For photos we
    also derive a brightness-dependent purple-fringe amount from the lens CA."""
    for p in presets.values():
        iso = p.setdefault("iso", 100)
        hp, ha, fp = iso_sensor_defaults(iso)
        p.setdefault("hot_px", hp)
        p.setdefault("hot_amt", ha)
        p.setdefault("fpn", fp)
        if kind == "photo":
            p.setdefault("fringe", round(min(0.5, p.get("ca", 0.0) * 200.0), 3))


PRESETS = {
    # Outdoor daylight: punchy CCD color, mild everything, warm auto-WB.
    "daylight": dict(
        mp=2.0, barrel=0.018, ca=0.0012, vignette=0.35,
        bloom_thresh=0.8, bloom_amt=0.5, bloom_sigma=6, smear=0.13,
        bayer=0.7, noise_lum=0.012, noise_chroma=0.05, chroma_nr=0.4,
        wb=(1.06, 1.0, 0.93), knee=0.85, sat=1.28, skin_magenta=0.5,
        contrast=0.22, black=0.02, sharpen=0.9, sharpen_sigma=1.0,
        jpeg_q=82, jpeg_passes=1, fmt="420",
    ),
    # The classic look: harsh on-camera flash. Hot center, dark falloff, cooler.
    "flash": dict(
        mp=2.0, barrel=0.018, ca=0.0013, vignette=0.6,
        bloom_thresh=0.7, bloom_amt=0.8, bloom_sigma=7, smear=0.16,
        bayer=0.7, noise_lum=0.02, noise_chroma=0.08, chroma_nr=0.45,
        wb=(1.02, 1.0, 1.02), knee=0.78, sat=1.18, skin_magenta=0.7,
        contrast=0.3, black=0.03, sharpen=1.0, sharpen_sigma=1.0,
        jpeg_q=78, jpeg_passes=1, fmt="420", flash=0.55,
    ),
    # Cheaper / older / higher-ISO indoor: more noise, softer, stronger artifacts.
    "lofi": dict(
        mp=1.0, barrel=0.03, ca=0.0018, vignette=0.5,
        bloom_thresh=0.78, bloom_amt=0.6, bloom_sigma=6, smear=0.24,
        bayer=0.9, noise_lum=0.035, noise_chroma=0.14, chroma_nr=0.5,
        wb=(1.05, 1.0, 0.9), knee=0.8, sat=1.15, skin_magenta=0.6,
        contrast=0.18, black=0.04, sharpen=1.2, sharpen_sigma=1.1,
        jpeg_q=68, jpeg_passes=2, fmt="420", iso=400,
    ),
    # Camcorder-still grab: low res, soft, heavy chroma loss.
    "camcorder": dict(
        mp=0.35, barrel=0.025, ca=0.0016, vignette=0.45,
        bloom_thresh=0.78, bloom_amt=0.6, bloom_sigma=5, smear=0.18,
        bayer=0.6, noise_lum=0.025, noise_chroma=0.12, chroma_nr=0.7,
        wb=(1.04, 1.0, 0.95), knee=0.8, sat=1.2, skin_magenta=0.5,
        contrast=0.2, black=0.03, sharpen=0.6, sharpen_sigma=1.2,
        jpeg_q=72, jpeg_passes=1, fmt="420", iso=200,
    ),

    # --- Camera profiles: modeled on documented signatures of real ~2002 digicams ---

    # Flagship "typical 2MP CCD digicam circa 2002": warm, punchy, balanced artifacts.
    "digicam": dict(
        mp=2.0, barrel=0.016, ca=0.0013, vignette=0.32,
        bloom_thresh=0.8, bloom_amt=0.5, bloom_sigma=6, smear=0.13,
        bayer=0.75, noise_lum=0.014, noise_chroma=0.05, chroma_nr=0.4,
        wb=(1.05, 1.0, 0.93), knee=0.84, sat=1.28, skin_magenta=0.5,
        contrast=0.22, black=0.02, sharpen=0.95, sharpen_sigma=1.0,
        jpeg_q=80, jpeg_passes=1, fmt="420",
    ),
    # Kodak DC / EasyShare ("Kodak Color Science"): warm, very saturated, punchy reds, soft.
    "kodak": dict(
        mp=2.1, barrel=0.018, ca=0.0012, vignette=0.35,
        bloom_thresh=0.8, bloom_amt=0.55, bloom_sigma=6, smear=0.12,
        bayer=0.8, noise_lum=0.012, noise_chroma=0.05, chroma_nr=0.45,
        wb=(1.09, 1.0, 0.89), knee=0.82, sat=1.42, skin_magenta=0.7,
        contrast=0.24, black=0.025, sharpen=0.8, sharpen_sigma=1.0,
        jpeg_q=80, jpeg_passes=1, fmt="420",
    ),
    # Sony Cyber-shot: fairly neutral / slightly cool, contrasty, heavily sharpened, clean.
    "sony": dict(
        mp=2.5, barrel=0.014, ca=0.0012, vignette=0.30,
        bloom_thresh=0.8, bloom_amt=0.45, bloom_sigma=5, smear=0.12,
        bayer=0.7, noise_lum=0.012, noise_chroma=0.045, chroma_nr=0.35,
        wb=(1.0, 1.0, 1.03), knee=0.80, sat=1.15, skin_magenta=0.3,
        contrast=0.30, black=0.03, sharpen=1.2, sharpen_sigma=1.0,
        jpeg_q=82, jpeg_passes=1, fmt="420",
    ),
    # Canon PowerShot: clean, slightly warm, balanced saturation, good detail, light artifacts.
    "canon": dict(
        mp=2.0, barrel=0.015, ca=0.0011, vignette=0.30,
        bloom_thresh=0.8, bloom_amt=0.45, bloom_sigma=6, smear=0.11,
        bayer=0.7, noise_lum=0.011, noise_chroma=0.04, chroma_nr=0.4,
        wb=(1.04, 1.0, 0.97), knee=0.83, sat=1.2, skin_magenta=0.45,
        contrast=0.22, black=0.02, sharpen=1.0, sharpen_sigma=1.0,
        jpeg_q=85, jpeg_passes=1, fmt="420",
    ),
    # Nikon Coolpix: crisp, slightly cool/green cast, a touch noisier.
    "nikon": dict(
        mp=2.0, barrel=0.016, ca=0.0014, vignette=0.32,
        bloom_thresh=0.8, bloom_amt=0.45, bloom_sigma=5, smear=0.12,
        bayer=0.75, noise_lum=0.016, noise_chroma=0.06, chroma_nr=0.4,
        wb=(0.99, 1.02, 1.0), knee=0.81, sat=1.12, skin_magenta=0.35,
        contrast=0.24, black=0.03, sharpen=1.1, sharpen_sigma=1.0,
        jpeg_q=80, jpeg_passes=1, fmt="420",
    ),
    # Fujifilm FinePix (Super CCD): vivid/velvia-like saturation, warm, smooth highlight roll-off.
    "fuji": dict(
        mp=2.5, barrel=0.015, ca=0.0012, vignette=0.30,
        bloom_thresh=0.82, bloom_amt=0.6, bloom_sigma=7, smear=0.1,
        bayer=0.7, noise_lum=0.012, noise_chroma=0.05, chroma_nr=0.4,
        wb=(1.05, 1.0, 0.95), knee=0.90, sat=1.4, skin_magenta=0.5,
        contrast=0.2, black=0.02, sharpen=0.9, sharpen_sigma=1.0,
        jpeg_q=82, jpeg_passes=1, fmt="420",
    ),

    # Night / long-exposure high-ISO grab: warm tungsten cast, lifted murky
    # blacks (AGC), heavy noise, strong smear off point lights, and the
    # low-light sensor tells -- hot/stuck pixels, readout stripe FPN, banding.
    "night": dict(
        mp=2.0, barrel=0.02, ca=0.0016, vignette=0.5,
        bloom_thresh=0.7, bloom_amt=0.85, bloom_sigma=8, smear=0.30,
        bayer=0.8, noise_lum=0.05, noise_chroma=0.18, chroma_nr=0.45,
        wb=(1.18, 1.0, 0.78), knee=0.82, sat=1.1, skin_magenta=0.5,
        contrast=0.16, black=0.06, sharpen=0.7, sharpen_sigma=1.1,
        jpeg_q=70, jpeg_passes=1, fmt="420", iso=800,
        hot_px=60, hot_amt=0.9, fpn=0.06, bands=72,
    ),
    # VHS still-frame grab: the "dubbed to tape" look on a single frame. Chroma
    # bandwidth collapses + bleeds (heavy chroma NR + chroma noise), luma softens
    # (low res, little sharpening), blacks lift to murk, colour desaturates with a
    # faint cast, and tracking / head-switch FPN stripes sit across the frame.
    "vhs": dict(
        mp=0.32, barrel=0.02, ca=0.0022, vignette=0.42,
        bloom_thresh=0.75, bloom_amt=0.5, bloom_sigma=7, smear=0.20,
        bayer=0.6, noise_lum=0.022, noise_chroma=0.09, chroma_nr=0.82,
        wb=(1.04, 1.02, 0.96), knee=0.86, sat=0.97, skin_magenta=0.4,
        contrast=0.16, black=0.025, sharpen=0.4, sharpen_sigma=1.3,
        jpeg_q=68, jpeg_passes=2, fmt="420", iso=200,
        hot_px=14, hot_amt=0.4, fpn=0.04, bands=60,
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
                          interlace=False, ca=0.003, vignette=0.45, noise=12, soft=0.5, mblur=3, bloom=0.25, smear=0.6,
                          sat=1.18, contrast=1.12, warm=0.06, sharp=0.6, chroma="yuvj420p",
                          arate=11025, ahp=250, alp=6000, abits=8, ahiss=0.005,
                          aagc="strong", adrive=2.0, acodec="adpcm_ima_wav"),
    # MiniDV camcorder: 720x480 interlaced 29.97fps, 4:1:1. Better audio: 32k 16-bit PCM.
    "camcorder": dict(w=720, h=480, fps=30000 / 1001, codec="dvvideo", qv=None, ext=".avi",
                      interlace=True, ca=0.0025, vignette=0.4, noise=20, soft=0.5, mblur=1, bloom=0.30, smear=0.6,
                      dr_curve="0/0.06 0.10/0.12 0.80/0.85 1/1",   # low-light AGC: lift shadows to noisy murk (not clean black)
                      sat=1.2, contrast=1.12, warm=0.05, sharp=0.5, chroma="yuv411p",
                      iso=400, hot_px=40, hot_amt=0.7, fpn=0.05,    # mild static defects (lifted shadows show them)
                      arate=32000, ahp=90, alp=13000, abits=0, ahiss=0.003,
                      aagc="med", adrive=0.0, acodec="pcm_s16le", zoom_motor=0.6),
    # Night / low-light CCD video: long integration time -> AGC gains up, so dark current
    # crosses threshold (static hot pixels), readout FPN shows in the lifted shadows, and
    # every light source blooms/smears hard. MJPEG 640x480 like a digicam movie mode.
    "night": dict(w=640, h=480, fps=15, codec="mjpeg", qv=10, ext=".avi",
                  interlace=False, ca=0.0035, vignette=0.55, noise=26, soft=0.6, mblur=4, bloom=0.5, smear=0.72,
                  dr_curve="0/0.05 0.10/0.15 0.74/0.88 1/1",   # lift shadows into noisy murk (AGC gain-up)
                  sat=1.05, contrast=1.05, warm=0.10, sharp=0.4, chroma="yuvj420p", iso=800,
                  hot_px=90, hot_amt=0.95, fpn=0.11,            # the low-light tells: defects + stripes
                  arate=11025, ahp=250, alp=6000, abits=8, ahiss=0.008,
                  aagc="strong", adrive=2.5, acodec="adpcm_ima_wav"),
    # VHS camcorder dubbed to tape: 640x480 interlaced, but recorded to VHS, so colour
    # bandwidth collapses + bleeds, luma softens, and a head-switch noise band sits at the
    # foot of the frame. Hi-fi-ish but wobbly audio (transport wow/flutter + hiss).
    "vhs": dict(w=640, h=480, fps=30000 / 1001, codec="mpeg4", qv=None, bitrate="1500k", ext=".avi",
                interlace=True, ca=0.0025, vignette=0.42, noise=18, soft=0.5, mblur=1, bloom=0.28, smear=0.55,
                dr_curve="0/0.05 0.10/0.12 0.80/0.86 1/1",
                sat=1.06, contrast=1.08, warm=0.04, sharp=0.35, chroma="yuv420p",
                iso=400, vhs=True, hot_px=25, hot_amt=0.6, fpn=0.04,
                arate=32000, ahp=80, alp=11000, abits=0, ahiss=0.005, awow=0.06,
                aagc="med", adrive=0.5, acodec="libmp3lame", abitrate="96k"),
    # Low-bitrate MPEG (early SD card / web clip): heavy macroblocking + warbly low-rate MP3.
    "mpeg_lofi": dict(w=320, h=240, fps=15, codec="mpeg4", qv=None, bitrate="320k", ext=".avi",
                      interlace=False, ca=0.004, vignette=0.5, noise=16, soft=0.6, mblur=3, bloom=0.25, smear=0.6,
                      sat=1.12, contrast=1.1, warm=0.07, sharp=0.5, chroma="yuv420p", iso=200,
                      arate=22050, ahp=200, alp=8000, abits=0, ahiss=0.004,
                      aagc="med", adrive=1.0, acodec="libmp3lame", abitrate="56k"),

    # --- Camera movie-mode profiles ---
    # Typical 2002 digicam movie mode (= digicam_video): MJPEG AVI + IMA-ADPCM.
    "digicam": dict(w=640, h=480, fps=15, codec="mjpeg", qv=8, ext=".avi",
                    interlace=False, ca=0.003, vignette=0.42, noise=12, soft=0.5, mblur=3, bloom=0.25, smear=0.6,
                    sat=1.22, contrast=1.12, warm=0.06, sharp=0.6, chroma="yuvj420p",
                    arate=11025, ahp=250, alp=6000, abits=8, ahiss=0.005,
                    aagc="strong", adrive=2.0, acodec="adpcm_ima_wav"),
    # Sony Cyber-shot "MPEG Movie": 320x240 @ ~16fps MPEG-1-style, tiny mono track.
    "sony": dict(w=320, h=240, fps=16, codec="mpeg4", qv=None, bitrate="550k", ext=".avi",
                 interlace=False, ca=0.003, vignette=0.4, noise=12, soft=0.5, mblur=3, bloom=0.22, smear=0.55,
                 sat=1.14, contrast=1.16, warm=0.0, sharp=0.7, chroma="yuv420p",
                 arate=16000, ahp=180, alp=7000, abits=0, ahiss=0.004,
                 aagc="med", adrive=1.0, acodec="libmp3lame", abitrate="64k"),
}

# Every preset now carries the ISO-scaled static sensor defects (hot pixels, FPN) and,
# for photos, a brightness-dependent purple-fringe amount -- faint at ISO 100, strong at
# ISO 800 -- so the sensor tells are always present and physically coupled, not bolted on
# to one preset. Presets that set values explicitly keep them.
_fill_sensor_defects(PRESETS, "photo")
_fill_sensor_defects(VIDEO_PRESETS, "video")


# Lighting / white-balance casts: the residual colour error a weak auto-WB left behind
# under a non-daylight source. These are per-channel linear gains (an illuminant tint),
# applied in the ISP white-balance stage exactly like the preset's own `wb`.
#   tungsten   : incandescent bulbs ~3200K; AWB under-corrects -> warm orange residue.
#   fluorescent: tubes spike in green; AWB can't fully null it -> green/cyan cast.
#   shade/blue : open shade ~7500K; AWB over-warms a cool scene -> cold blue residue.
CAST_MULT = {
    "tungsten":    (1.12, 1.00, 0.82),
    "fluorescent": (0.95, 1.07, 0.97),
    "shade":       (0.90, 0.98, 1.14),
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


def process_photo(in_path, out_path, p, datestamp=None, strength=1.0, seed=None,
                  smear_mode="classic", cast=None):
    img = Image.open(in_path).convert("RGB")
    W0, H0 = img.size

    x = np.asarray(img, np.float32) / 255.0
    lin = srgb_to_linear(x)
    seed, _ = _resolve_seed(seed)
    rng = np.random.default_rng(seed)                    # fresh thermal noise each render
    sensor_rng = np.random.default_rng(SENSOR_SEED)      # fixed per-camera defect map

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
    lin = ccd_smear(lin, clip, amt(p["smear"]), smear_mode)
    lin = purple_fringe(lin, clip, amt(p.get("fringe", 0.0)))             # brightness-dependent CA fringe
    lin = highlight_knee(lin, p["knee"])                                  # full-well roll-off (pre-WB)
    lin = add_noise(lin, amt(p["noise_lum"]), amt(p["noise_chroma"]), rng)         # varies per render
    lin = fixed_pattern_noise(lin, sensor_rng, amt(p.get("fpn", 0.0)))             # fixed readout stripe pattern
    lin = hot_pixels(lin, sensor_rng, p.get("hot_px", 0.0), amt(p.get("hot_amt", 0.0)))  # fixed stuck/hot sensels
    lin = bayer_emulate(lin, p["bayer"] * strength)

    # --- (e) ISP: white balance, then display-referred color/tone/NR/sharpen ---
    lin = lin * np.array(p["wb"], np.float32)
    if cast in CAST_MULT:                                  # residual illuminant cast (weak AWB)
        lin = lin * np.array(CAST_MULT[cast], np.float32)
    x = linear_to_srgb(lin)
    x = saturate(x, 1 + (p["sat"] - 1) * strength, p["skin_magenta"])
    x = s_curve(x, p["contrast"] * strength, p["black"])
    x = chroma_nr(x, amt(p["chroma_nr"]))
    x = unsharp(x, amt(p["sharpen"]), p["sharpen_sigma"])
    x = tone_banding(x, p.get("bands", 0))                                # 8-bit gradient banding

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


def video_smear(in_lbl, out_lbl, amount, h, mode="classic"):
    """CCD vertical smear for video. Highlights are found by relative brightness
    ('prominence' = luma minus a local average), spread down the column, tinted and
    blended over the frame.

    Two models:
      * "classic" (default): a uniform magenta/purple streak, screen-blended. The
        original look; reads clearly.
      * "physical" (WIP): additive charge whose tint washes from purple (faint) to white
        (bright). Still being tuned (it can look too strong or too weak by scene).
    """
    bg = max(24, h // 3)                                   # local-average radius (relative-brightness gate)
    sy = max(20, h // 2)                                   # vertical reach
    pre = (f"[{in_lbl}]split=3[sm0][smA][smB];"
           f"[smB]format=gray,gblur=sigma={bg}[smavg];"
           f"[smA]format=gray[smg];"
           f"[smg][smavg]blend=all_mode=subtract[smc];"    # prominence = luma - local avg
           f"[smc]curves=all='0/0 0.08/0 0.30/1 1/1',avgblur=sizeX=1:sizeY={sy}[q];")
    if mode == "physical":
        g = amount * 1.4
        return (pre +
                f"[q]split[qa][qb];"
                f"[qb]curves=all='0/0 0.5/0.32 0.85/0.7 1/1'[qg];"     # green lags -> purple..white
                f"[qg][qa]mergeplanes=0x001010:gbrp,"
                f"colorchannelmixer=rr={g:.3f}:gg={g:.3f}:bb={g:.3f}[smt];"
                f"[sm0][smt]blend=all_mode=addition[{out_lbl}]")
    return (pre +                                          # classic uniform purple, screen-blended
            f"[q]format=gbrp,colorchannelmixer=rr=0.95:gg=0.32:bb=1.08[smt];"
            f"[sm0][smt]blend=all_mode=screen:all_opacity={amount}[{out_lbl}]")


def _hot_overlay_img(path, w, h, density, amount, seed):
    """Render the static hot/stuck-pixel defect map as a black RGB PNG (bright dots
    on black) to be ADDED over every video frame (blend=addition).

    The whole point is staticness: the same sensels are defective in every frame, so
    the defects sit in fixed positions, exactly like a given CCD's permanent defect
    map. Per-frame moving 'noise' (added later) can never reproduce that, and the
    fixed dots are the strongest tell that footage came off a real sensor. Most
    visible in the shadows (the added charge is swamped by signal in the highlights)."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((h, w, 3), np.float32)
    n = max(1, int(round(density * (h * w) / 1e6)))
    ys = rng.integers(0, h, n); xs = rng.integers(0, w, n)
    mag = np.clip(amount * (0.5 + rng.random(n)), 0, 1).astype(np.float32)
    ch = rng.integers(0, 3, n)
    stuck = rng.random(n) < 0.6                            # 60% single-channel, 40% white-hot
    for k in range(n):
        if stuck[k]:
            canvas[ys[k], xs[k], ch[k]] = mag[k]
        else:
            canvas[ys[k], xs[k], :] = mag[k]
    Image.fromarray((canvas * 255 + 0.5).astype(np.uint8), "RGB").save(path)


def _fpn_overlay_img(path, w, h, amount, seed):
    """Render the static fixed-pattern-noise stripe map as a mid-gray (128) RGB PNG.

    Per-column offset (the dominant CCD term: each column clocks out through its own
    path) plus a weaker per-row term, centered on neutral gray so blend=grainmerge
    adds it (out = frame + dev). Frame-constant, so it reads as a stationary stripe
    texture in flat shadows/skies, the way real readout FPN does, not as grain."""
    rng = np.random.default_rng(seed + 1)
    col = rng.standard_normal(w).astype(np.float32) * amount * 14.0
    row = rng.standard_normal(h).astype(np.float32) * amount * 4.0
    dev = col[None, :] + row[:, None]
    gray = np.clip(128.0 + dev, 0, 255)
    Image.fromarray(np.repeat(gray[:, :, None], 3, 2).astype(np.uint8), "RGB").save(path)


def _hsw_band(h):
    """Head-switch band height: a thin strip at the very foot of the frame. Real head
    switching sits in the few lines of the vertical-blanking interval below the active
    picture, hidden by TV overscan and only seen in a full-raster capture, so it is a
    thin bottom band, not a tall one."""
    return max(4, int(round(h * 0.03)))


def _headswitch_img(path, w, h):
    """Render the alpha ENVELOPE of the VHS head-switching band: a grayscale strip that
    is transparent at the top and opaque at the foot.

    On VHS the two video heads hand off near the bottom of each field, and the few lines
    around the switch point can't be tracked, so they fill with the noise the head reads
    off un-recorded tape. That noise is different every frame, so it is generated live in
    the filtergraph (see build_vhs_stmts); this image only supplies the fixed shape, a
    soft top edge ramping into a solid band at the picture's foot, used as its alpha."""
    band = _hsw_band(h)
    ramp = np.clip(np.linspace(0.0, 1.0, band) * 1.7, 0, 1)           # fade in from the top edge
    img = np.broadcast_to(ramp[:, None] * 255.0, (band, w)).astype(np.uint8)
    Image.fromarray(img, "L").save(path)


def build_vhs_stmts(in_lbl, out_lbl, w, h, fps, hsw_png=None, hsw_seed=-1):
    """VHS tape playback degradation, applied AFTER the camera ISP/overlay (the tape records
    a finished signal, then a worn deck plays it back). Modelled on the real playback signal
    chain, IN ORDER:

      1. MECHANICAL READ / time-base error. The helical-scan heads and the capstan move the
         tape past the gap with imperfect timing, so each scan line is read a touch early or
         late -> a per-line HORIZONTAL position error, applied with the `displace` filter
         from a row-dependent displacement map. Three real components:
           * wow + flutter: continuous, low-amplitude drift from drum/capstan/roller
             eccentricity (a couple of Hz plus tens of Hz) -- subtle, always there,
           * flagging: the servo hasn't settled just after vertical sync, so the error is
             largest at the TOP of the field and decays down it ("flag-waving"),
           * tracking disturbances: ~every 10 s the head wanders off the track for ~a second
             -- the time-base error spikes (picture jitters, the top tears) and the off-track
             read throws a noise band (step 5) AT THE SAME MOMENT (shared envelope).
      2. color-under chroma: bandwidth collapse + Y/C delay + chroma noise; luma softness.
      3. dropouts: brief bright streaks where the tape momentarily loses head contact.
      4. head-switching: a thin band of off-tape hash in the overscan at the field foot.

    The dramatic errors are intermittent; between disturbances only the subtle continuous
    flutter remains. Tape noise shares the per-render hsw_seed. Returns stmts ending [out_lbl]."""
    cw = max(2, w // 11)                                    # collapsed chroma width (~1/11 luma)
    band = _hsw_band(h)
    r = f"{fps:.4f}"
    # disturbance envelope: ~0 most of the time, a smooth pulse ~every 10s (~1s wide). Shared by
    # the time-base spike (dx, below) and the tracking band (step 5) so the jitter and the band
    # happen together. egeq uses geq's uppercase T; etl uses the timeline's lowercase t.
    egeq = "pow(0.5+0.5*sin(2*PI*T/10)\\,8)"
    etl  = "pow(0.5+0.5*sin(2*PI*t/10)\\,8)"
    y0 = f"({h}*0.09)"                                      # flagging decay scale (top ~9% of lines)
    dx = ("128"                                             # per-line x-displacement (128 = no shift)
          "+1.3*sin(2*PI*1.7*T)+0.7*sin(2*PI*9*T)"          # wow + flutter (always, subtle)
          f"+(1.5+7*{egeq})*exp(-Y/{y0})"                   # flagging (top of field; spikes on disturbance)
          f"+5*{egeq}*sin(2*PI*6*T)")                       # whole-frame jitter (disturbance only)
    s = [
        # 1. MECHANICAL READ time-base error: heads/capstan read each line with imperfect timing
        #    -> per-line horizontal position error. Build a row-dependent X-displacement map
        #    (cheap 32-wide geq stretched to width) + a flat Y-map (no vertical shift), displace.
        f"color=c=gray:s=32x{h}:r={r}:d=3600,format=gray,geq=lum='{dx}',"
        f"scale={w}:{h}:flags=bilinear,format=rgb24,setsar=1[xmap];"
        f"color=c=gray:s={w}x{h}:r={r}:d=3600,format=rgb24,setsar=1[ymap];"
        f"[{in_lbl}]format=rgb24,setsar=1[vin];"
        f"[vin][xmap][ymap]displace=edge=smear[vmech]",
        # 2. color-under chroma: bandwidth collapse + Y/C delay + chroma noise + luma softness
        f"[vmech]format=yuv444p,extractplanes=y+u+v[vy][vu][vv];"
        f"[vu]scale={cw}:{h}:flags=bilinear,scale={w}:{h}:flags=bilinear,setsar=1[vus];"
        f"[vv]scale={cw}:{h}:flags=bilinear,scale={w}:{h}:flags=bilinear,setsar=1[vvs];"
        f"[vy]gblur=sigma=0.8:sigmaV=0,setsar=1[vys];"
        f"[vys][vus][vvs]mergeplanes=0x001020:yuv444p,"
        f"chromashift=crh=5:cbh=-4,"                        # Y/C delay (colour lags luma)
        f"noise=c1s=24:c2s=24:allf=t+u:all_seed={hsw_seed},"  # temporal chroma noise
        f"format=rgb24,setsar=1[vtape]"]
    cur = "vtape"
    # tape dropouts: rare bright horizontal dashes, fresh every frame
    s.append(
        f"color=c=black:s={w}x{h}:r={r}:d=3600,format=yuv444p,"
        f"noise=alls=100:allf=t+u:all_seed={hsw_seed + 1},"
        f"lutyuv=y='if(gt(val\\,252)\\,255\\,0)':u=128:v=128,"  # keep only the rare peaks -> sparse specks
        f"scale={max(2, w // 45)}:{h}:flags=bilinear,scale={w}:{h}:flags=bilinear,"  # stretch into streaks
        f"format=gbrp,setsar=1[drp];"
        f"[{cur}][drp]blend=all_mode=screen:shortest=1[vdrop]")
    cur = "vdrop"
    # (time-base error is modelled up front in step 1 via displace, not here)
    if hsw_png:
        s.append(
            f"color=c=gray:s={w}x{band}:r={r}:d=3600,"
            f"noise=alls=100:allf=t+u:all_seed={hsw_seed + 2},"
            f"eq=contrast=2.2,format=gbrp,setsar=1[hnz];"   # harsh off-tape hash (noise caps at 100)
            f"movie='{hsw_png}':loop=0,setpts=N/(FRAME_RATE*TB),format=gray,setsar=1[hramp];"
            f"[hnz][hramp]alphamerge[hband];"
            f"[{cur}][hband]overlay=0:H-h:shortest=1[vhsw]")
        cur = "vhsw"
    else:
        s.append(f"[{cur}]copy[vhsw]")
        cur = "vhsw"
    # 5. tracking disturbance: when the head wanders off the track (SAME envelope as the
    #    time-base spike, so the jitter and the band coincide), the off-track read throws a
    #    snowy band low in the frame, by the head-switch point. Present only in those windows.
    trkH = max(10, int(round(h * 0.08)))
    s.append(
        f"color=c=gray:s={w}x{trkH}:r={r}:d=3600,"
        f"noise=alls=100:allf=t+u:all_seed={hsw_seed + 3},eq=contrast=2.4,format=gbrp,setsar=1[trk];"
        f"[{cur}][trk]overlay=0:{int(h - band - trkH)}:enable=gt({etl}\\,0.35):shortest=1[{out_lbl}]")
    return s


def _osd_font():
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    return f":fontfile={font}" if os.path.exists(font) else ""


def build_osd_filters(vp):
    """Burnt-in camcorder on-screen display (the chrome a camera drew over the picture
    while recording, so it lives in the scan/overlay stage with the date stamp):
      * a blinking red dot + REC top-left,
      * a running timecode bottom-left,
      * a battery gauge top-right (outline + terminal + charge bar),
      * an orange clock under the battery.
    All sized off the frame height so it scales across 320/640/720-wide presets."""
    w, h = vp["w"], vp["h"]
    u = h / 480.0                                          # scale unit (designed at 480 tall)
    fs = max(12, int(h * 0.055))
    cfs = max(11, int(fs * 0.58))                          # clock runs smaller than REC/timecode
    fo = _osd_font()
    sh = ":shadowcolor=black:shadowx=2:shadowy=2"
    pad = int(24 * u)
    bw = int(34 * u); bh = int(15 * u); bx = int(90 * u); by = int(24 * u)   # compact battery (inset from edge so tape overscan/wobble doesn't clip it)
    rpad = int(48 * u)                                     # right-side inset for battery+clock
    rate = max(1, int(round(vp["fps"])))
    return [
        # blinking REC (on for the first half of each second)
        f"drawtext=text='● REC'{fo}:fontcolor=red:fontsize={fs}:x={pad}:y={pad}"
        f"{sh}:enable='lt(mod(t\\,1)\\,0.5)'",
        # running timecode, bottom-left
        f"drawtext=timecode='00\\:00\\:00\\:00':rate={rate}{fo}:fontcolor=white:fontsize={fs}"
        f":x={pad}:y=h-th-{pad}{sh}",
        # battery: outline body, positive terminal, then a charge bar inside
        f"drawbox=x=iw-{bx}:y={by}:w={bw}:h={bh}:color=white:t={max(1, int(1.5 * u))}",
        f"drawbox=x=iw-{bx - bw}:y={by + int(bh * 0.3)}:w={max(2, int(3 * u))}:h={int(bh * 0.4)}:color=white:t=fill",
        f"drawbox=x=iw-{bx - int(3 * u)}:y={by + int(2.5 * u)}:w={int(bw * 0.62)}:h={bh - int(5 * u)}:color=white@0.85:t=fill",
        # running wall clock: starts 03:00 AM and advances with the recording (gmtime base
        # 97200 = 03:00:00; pts is added each frame). under the battery, right-aligned.
        f"drawtext=text='%{{pts\\:gmtime\\:97200\\:%I\\\\\\:%M %p}}'{fo}:fontcolor=orange:fontsize={cfs}"
        f":x=w-tw-{rpad}:y={by + bh + int(8 * u)}{sh}",
    ]


def build_post_filters(vp, datestamp, noise_seed=-1):
    """Filters AFTER bloom/smear, in imaging order:
        sensor (limited DR -> read noise) -> ISP (white balance -> sat/contrast ->
        sharpen) -> scan/overlay (interlace, datestamp, OSD).
    Vignetting is optical and is done back in the geometry stage, not here.
    `noise_seed` seeds the read-noise grain so it differs each render (it is thermal,
    not a sensor defect); -1 lets ffmpeg pick its default."""
    h = vp["h"]
    cm = CAST_MULT.get(vp.get("cast"))                                   # residual illuminant cast
    wb = f"colorbalance=rm={vp['warm']}:bm=-{vp['warm']}"                # white balance first
    if cm:
        wb += f",colorchannelmixer=rr={cm[0]}:gg={cm[1]}:bb={cm[2]}"     # then the illuminant tint
    filters = [
        # --- sensor response ---
        # limited CCD dynamic range. Default crushes shadows + rolls highlights to kill
        # the flat modern look. Camcorders in low light instead *lift* the shadows
        # (AGC gain-up) into noisy murk -> presets can override `dr_curve`.
        f"curves=master='{vp.get('dr_curve', '0/0 0.12/0.05 0.78/0.86 1/1')}'",
        f"noise=alls={vp['noise']}:allf=t+u:all_seed={noise_seed}",     # read noise (before ISP sharpen)
        # --- ISP ---
        wb,
        f"eq=saturation={vp['sat']}:contrast={vp['contrast']}",         # then color matrix / tone
        f"unsharp=5:5:{vp['sharp']}:5:5:0.0",                            # sharpening (amplifies the noise above)
    ]
    if vp.get("interlace"):
        filters.append("interlace=scan=tff:lowpass=1")                  # combing on motion
    if datestamp:
        fontopt = _osd_font()
        filters.append(
            f"drawtext=text='{datestamp}'{fontopt}:fontcolor=orange:fontsize={max(14,int(h*0.05))}"
            f":x=w-tw-12:y=h-th-12:shadowcolor=0x802000:shadowx=1:shadowy=1")
    if vp.get("osd"):
        filters += build_osd_filters(vp)                                # camcorder OSD chrome
    return ",".join(filters)


def build_video_graph(vp, datestamp, smear_mode="classic", hot_png=None, fpn_png=None,
                      hsw_png=None, noise_seed=-1, hsw_seed=-1):
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
        stmts.append(video_smear(cur, "smr", vp["smear"], h, smear_mode)); cur = "smr"

    # --- sensor: STATIC defects -- fixed-pattern stripe noise + hot/stuck pixels ---
    # Both are frame-constant (same pixels every frame), unlike the per-frame read noise
    # added in the ISP block below. That staticness is the real CCD tell. The maps are
    # generated once as PNGs and looped in via the movie source (no extra -i input):
    # FPN is mid-gray and grainmerge-added (frame + dev); hot dots are black + addition.
    if hot_png or fpn_png:
        stmts.append(f"[{cur}]format=rgb24,setsar=1[sbase]"); cur = "sbase"
        if fpn_png:
            stmts.append(f"movie='{fpn_png}':loop=0,setpts=N/(FRAME_RATE*TB),format=rgb24,setsar=1[fpnov];"
                         f"[{cur}][fpnov]blend=all_mode=grainmerge:shortest=1[fpnd]"); cur = "fpnd"
        if hot_png:
            stmts.append(f"movie='{hot_png}':loop=0,setpts=N/(FRAME_RATE*TB),format=rgb24,setsar=1[hotov];"
                         f"[{cur}][hotov]blend=all_mode=addition:shortest=1[hotd]"); cur = "hotd"

    # --- sensor response + ISP + scan/overlay ---
    # If a VHS stage follows, the ISP output is the signal "sent to the recorder", so
    # post filters land on an intermediate label and the tape degradation produces [v].
    if vp.get("vhs"):
        stmts.append(f"[{cur}]{build_post_filters(vp, datestamp, noise_seed)}[pf]")
        stmts += build_vhs_stmts("pf", "v", w, h, fps, hsw_png, hsw_seed)
    else:
        stmts.append(f"[{cur}]{build_post_filters(vp, datestamp, noise_seed)}[v]")
    return ";".join(stmts)


def build_audio_graph(vp, motor=False, hiss_seed=-1):
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
    hiss = (f"anoisesrc=color=pink:amplitude={vp['ahiss']}:sample_rate=48000:seed={hiss_seed},"
            "aformat=channel_layouts=mono[hs]")                          # mic/circuit self-noise
    pre = [mono, hiss]
    if motor:
        pre.append("[1:a]aformat=channel_layouts=mono[mo]")              # zoom-motor whir (input 1)
        mix = "[m][mo][hs]amix=inputs=3:duration=first:dropout_transition=0:normalize=0[mix]"
    else:
        mix = "[m][hs]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]"
    chain = [f"highpass=f={vp['ahp']}", f"lowpass=f={vp['alp']}"]         # mic band (no bass/highs)
    if vp.get("awow", 0) > 0:
        chain.append(f"vibrato=f=5:d={vp['awow']}")                       # tape transport wow/flutter
    chain.append(AGC_COMPAND[vp["aagc"]])                                 # AGC pumps signal+hiss
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


def process_video(in_path, out_path, vp, datestamp=None, audio=True, motor_wav=None,
                  smear_mode="classic", seed=None):
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (needed for video).")
    do_audio = audio and has_audio(in_path)
    use_motor = bool(motor_wav) and do_audio

    # Per-render seeds for the thermal noise (read grain, tape head-switch hash, mic hiss)
    # so they differ each render; the sensor defect maps stay on the fixed SENSOR_SEED.
    # Keep sub-seeds well under INT_MAX: the VHS stage derives hsw_seed+1/+2 for its
    # extra noise sources, and ffmpeg's noise all_seed caps at INT_MAX (2^31-1).
    _, sub = _resolve_seed(seed)
    noise_seed = sub.randrange(1 << 30)
    hsw_seed = sub.randrange(1 << 30)
    hiss_seed = sub.randrange(1 << 30)

    # Generate the static sensor-defect maps once (looped in by build_video_graph). Hot
    # pixels and FPN are the camera's permanent defect map -> fixed SENSOR_SEED. The VHS
    # head-switch band is tape playback noise, not a sensor defect -> fresh seed.
    hot_png = fpn_png = None
    tmp_pngs = []
    if vp.get("hot_px"):
        hot_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        _hot_overlay_img(hot_png, vp["w"], vp["h"], vp["hot_px"], vp.get("hot_amt", 0.8), seed=SENSOR_SEED)
        tmp_pngs.append(hot_png)
    if vp.get("fpn"):
        fpn_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        _fpn_overlay_img(fpn_png, vp["w"], vp["h"], vp["fpn"], seed=SENSOR_SEED)
        tmp_pngs.append(fpn_png)
    hsw_png = None
    if vp.get("vhs"):
        hsw_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        _headswitch_img(hsw_png, vp["w"], vp["h"])
        tmp_pngs.append(hsw_png)

    try:
        graph = build_video_graph(vp, datestamp, smear_mode, hot_png, fpn_png, hsw_png,
                                  noise_seed, hsw_seed)
        if do_audio:
            graph += ";" + build_audio_graph(vp, motor=use_motor, hiss_seed=hiss_seed)

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
    finally:
        for t in tmp_pngs:
            try:
                os.unlink(t)
            except OSError:
                pass


def audio_ext(vp):
    """File extension for an audio-only output, matching the preset's codec."""
    return ".mp3" if vp["acodec"] == "libmp3lame" else ".wav"


def process_audio(in_path, out_path, vp, seed=None):
    """Audio-only: run a source sound through the built-in-mic degradation chain
    (mono -> hiss -> band-limit -> AGC -> ADC bit-crush/sample-rate) and encode with
    the preset's period codec. Same chain used for a video's soundtrack."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH (needed for audio).")
    _, sub = _resolve_seed(seed)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", in_path,
           "-filter_complex", build_audio_graph(vp, hiss_seed=sub.randrange(1 << 31)),
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

def _phase_shift(a, b):
    """(dx, dy, confidence) that shifts `a` onto `b`, by phase correlation. The cross-power
    spectrum depends only on translation, so this is a robust translation estimator; the
    normalized correlation peak height is returned as a confidence."""
    A = np.fft.rfft2(a); B = np.fft.rfft2(b)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-6
    c = np.fft.irfft2(R, s=a.shape)
    idx = int(np.argmax(c)); peak = float(c.flat[idx])
    dy, dx = np.unravel_index(idx, c.shape)
    H, W = a.shape
    if dy > H // 2: dy -= H
    if dx > W // 2: dx -= W
    return int(dx), int(dy), peak


def _scale_about_center(img, c):
    """Scale a 2D frame by factor c about its center, same output size."""
    H, W = img.shape
    nw, nh = max(1, round(W * c)), max(1, round(H * c))
    im = Image.fromarray(img.astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (W, H))
    canvas.paste(im, ((W - nw) // 2, (H - nh) // 2))
    return np.asarray(canvas, np.float32)


def detect_zoom(in_path, analyze_fps=5):
    """Return (start_s, end_s, speed) zoom segments.

    A zoom is a SCALE about the center; handheld motion is mostly TRANSLATION. For each
    frame pair we estimate and remove the translation (phase correlation), then find the
    scale-about-center that best matches the next frame and how much better it fits than
    no-scaling (ratio). Because a real *optical* zoom is deliberate, we only keep a
    segment when the scale change is strong, sustained, and (unlike shake or a dolly
    wobble) monotonic in one direction. Note a strong forward dolly scales the image like
    a zoom and cannot be told apart from 2D motion alone, so it may also register.
    """
    W, H = 160, 90
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", in_path, "-vf",
         f"fps={analyze_fps},scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    n = len(raw) // (W * H)
    if n < 6:
        return []
    frames = np.frombuffer(raw[:n * W * H], np.uint8).reshape(n, H, W).astype(np.float32)
    cand = np.linspace(0.94, 1.06, 7)                        # inter-frame scale guesses (index 3 == 1.0)
    mid = 3
    y0, y1, x0, x1 = int(H * 0.3), int(H * 0.7), int(W * 0.3), int(W * 0.7)
    best = np.ones(n - 1, np.float32)
    ratio = np.ones(n - 1, np.float32)
    for i in range(n - 1):
        dx, dy, _ = _phase_shift(frames[i], frames[i + 1])   # remove handheld translation
        a = np.roll(np.roll(frames[i], dy, 0), dx, 1)
        b = frames[i + 1][y0:y1, x0:x1]
        err = [np.mean((_scale_about_center(a, c)[y0:y1, x0:x1] - b) ** 2) for c in cand]
        j = int(np.argmin(err))
        best[i] = cand[j]
        ratio[i] = err[j] / (err[mid] + 1e-6)
    if len(best) >= 5:                                       # smooth out jitter (edge-padded so a
        best = np.convolve(np.pad(best, 2, mode="edge"), np.ones(5) / 5, mode="valid")  # zoom at t=0 survives
    rate = np.log(np.clip(best, 1e-3, None))                 # >0 zoom in, <0 zoom out
    zooming = (ratio < 0.8) & (np.abs(rate) > np.log(1.012))  # scaling must genuinely help
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
        seg = rate[a:b + 1]
        if (b - a + 1) * dt < 0.8:                           # deliberate zooms last a moment
            continue
        net = float(np.sum(seg))
        if abs(net) < np.log(1.20):                          # < ~20% total -> drift/dolly wobble, not a zoom
            continue
        if np.mean(np.sign(seg) == np.sign(net)) < 0.75:     # must be monotonic, not back-and-forth shake
            continue
        out.append((a * dt, (b + 1) * dt, abs(net) / ((b - a + 1) * dt)))
    return out


def _movavg(x, k):
    """Simple box moving average (used as a crude low-pass on 1D audio)."""
    if k < 2:
        return x
    return np.convolve(x, np.ones(k, np.float32) / k, mode="same")


def synth_motor_track(total_dur, intervals, sr=48000, intensity=0.5, seed=7):
    """Synthesize a mono zoom-motor track.

    A real lens zoom motor is a steady LOW-PITCH mechanical whir, not a musical note, so
    this is mostly band-limited mechanical noise (a soft 'vvvvv') with only a faint low
    hum and slow gear roughness on top, and long smooth fades so there is no 'boing'. The
    mic chain band-limits and AGC-pumps it downstream like a real built-in mic.
    """
    rng = np.random.default_rng(seed)
    track = np.zeros(int(total_dur * sr) + sr, np.float32)
    for t0, t1, speed in intervals:
        n = int((t1 - t0) * sr)
        if n <= 0:
            continue
        t = np.arange(n) / sr
        # broadband mechanical whir: noise band-limited to a low band (~150-1500 Hz)
        noise = rng.standard_normal(n).astype(np.float32)
        low = _movavg(noise, 16)                              # low-pass ~1.5 kHz
        whir = low - _movavg(low, 120)                        # remove rumble -> low/mid band
        s = whir.std()
        if s > 1e-6:
            whir = whir / s
        # faint, non-musical low hum (kept quiet so it is not a tone)
        f0 = 90.0 + 20.0 * float(np.clip(speed, 0, 3))
        hum = 0.18 * np.sin(2 * np.pi * f0 * t) + 0.08 * np.sin(2 * np.pi * 2.0 * f0 * t)
        # slow random amplitude wobble (gear/load variation), NOT pitch vibrato
        am = 1.0 + 0.3 * _movavg(rng.standard_normal(n).astype(np.float32), 400)
        sig = (0.85 * whir + 0.35 * hum) * am
        env = np.ones(n, np.float32)                          # long smooth fades (no onset thump)
        a = min(n, int(0.18 * sr)); r = min(n, int(0.22 * sr))
        env[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
        env[-r:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r))
        sig *= env * intensity * 0.6
        st = int(t0 * sr)
        m = min(n, len(track) - st)
        track[st:st + m] += sig[:m]
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
    "vhs": "VHS still grab: bled chroma, soft, murky, tracking stripes",
    "night": "long-exposure low-light: warm, noisy, hot pixels + banding",
}
VIDEO_DESC = {
    "digicam": "digicam movie mode, MJPEG 640x480 + IMA-ADPCM",
    "digicam_video": "same as digicam",
    "sony": "Cyber-shot MPEG Movie, 320x240 @16fps",
    "camcorder": "MiniDV, interlaced 720x480, low-light grain",
    "mpeg_lofi": "low-bitrate MPEG-4 320x240, heavy macroblocking",
    "night": "low-light CCD video, lifted murk, static hot pixels + FPN",
    "vhs": "camcorder dubbed to VHS: chroma bleed, head-switch noise",
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
    zoom: Optional[str] = typer.Option(None, "--zoom",
                                       help="Add a zoom-motor whir over time ranges, e.g. '2-4,7-8.5' (seconds), video."),
    zoom_sound: bool = typer.Option(False, "--zoom-sound",
                                    help="[WIP] also try to auto-detect zoom (unreliable; see README)."),
    smear: str = typer.Option("classic", "--smear",
                              help="CCD smear model: 'classic' (default) or 'physical' (WIP)."),
    osd: bool = typer.Option(False, "--osd",
                             help="Burn in a camcorder OSD: blinking REC, timecode, battery, clock (video)."),
    cast: Optional[str] = typer.Option(None, "--cast",
                                       help="Light/WB cast: tungsten, fluorescent, or shade (photo + video)."),
    seed: Optional[int] = typer.Option(None, "--seed",
                                       help="Pin the random-noise seed for a reproducible render "
                                            "(default: fresh each run). Sensor defects use a fixed seed regardless."),
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
    if cast is not None and cast not in CAST_MULT:
        typer.secho(f"Error: unknown --cast '{cast}'. Choose from: {', '.join(CAST_MULT)}.",
                    fg="red", err=True)
        raise typer.Exit(2)

    ext = input.suffix.lower()
    if ext in VID_EXT:
        if preset not in VIDEO_PRESETS:
            typer.secho(f"note: '{preset}' has no video profile; using 'digicam_video'.", fg="yellow")
        vp = dict(VIDEO_PRESETS.get(preset, VIDEO_PRESETS["digicam_video"]))
        if osd:
            vp["osd"] = True
        if cast:
            vp["cast"] = cast
        out = str(output) if output else str(input.with_suffix("")) + ".digicam" + vp["ext"]
        motor_wav = None
        if not no_audio and has_audio(str(input)):
            segs = []
            if zoom:                                          # manual time ranges override detection
                for part in zoom.split(","):
                    if "-" in part:
                        a, b = part.split("-", 1)
                        segs.append((float(a), float(b), 0.5))
            elif zoom_sound:
                with console.status("[cyan]scanning for zoom[/]", spinner="dots"):
                    segs = detect_zoom(str(input))
            if segs:
                total = sum(b - a for a, b, _ in segs)
                console.print(f"[cyan]zoom[/] {len(segs)} segment(s), {total:.1f}s -> adding motor sound")
                motor_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                write_wav(motor_wav, synth_motor_track(media_duration(str(input)), segs,
                                                       intensity=vp.get("zoom_motor", 0.35)))
        try:
            process_video(str(input), out, vp, datestamp, audio=not no_audio,
                          motor_wav=motor_wav, smear_mode=smear, seed=seed)
        finally:
            if motor_wav:
                os.unlink(motor_wav)
    elif ext in AUD_EXT:
        if preset not in VIDEO_PRESETS:
            typer.secho(f"note: '{preset}' has no audio profile; using 'digicam_video'.", fg="yellow")
        vp = dict(VIDEO_PRESETS.get(preset, VIDEO_PRESETS["digicam_video"]))
        out = str(output) if output else str(input.with_suffix("")) + ".digicam" + audio_ext(vp)
        process_audio(str(input), out, vp, seed=seed)
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
            w, h = process_photo(str(input), out, p, datestamp, strength, seed,
                                 smear_mode=smear, cast=cast)
        console.print(f"[green]wrote[/] {out}  [dim]{w}x{h}[/]")
        return
    else:
        typer.secho(f"Error: unsupported file type '{ext}'.", fg="red", err=True)
        raise typer.Exit(2)
    console.print(f"[green]wrote[/] {out}")


if __name__ == "__main__":
    app()
