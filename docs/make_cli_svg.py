#!/usr/bin/env python3
"""Regenerate docs/cli.svg: a representative digicam2000 terminal session.

Run with the project on the path so it reflects the live presets/flags:
    PYTHONPATH=. python docs/make_cli_svg.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console, Group
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

import digicam2000 as d

console = Console(record=True, width=92)


def prompt(cmd):
    console.print(f"[bold green]$[/] [bold white]digicam2000[/] [cyan]{cmd}[/]")


# 1) the preset listing (borderless: box-drawing glyphs rasterize as tofu in cairosvg)
prompt("--list")
console.print()
for title, names, desc in (("Photo presets", d.PRESETS, d.PRESET_DESC),
                           ("Video / audio presets", d.VIDEO_PRESETS, d.VIDEO_DESC)):
    tbl = d._preset_table(title, names, desc)
    tbl.box = None
    console.print(tbl)
    console.print()

# 2) a photo run
prompt("beach.jpg -p kodak --cast tungsten")
console.print("[green]wrote[/] beach.digicam.jpg  [dim]1600x1200[/]")
console.print()

# 3) a video run, with a finished progress bar captured
prompt("trip.mov -p vhs -d 2002-07-04 --osd")
with Progress(TextColumn("[bold cyan]{task.description}"), BarColumn(bar_width=None),
              TextColumn("{task.percentage:>3.0f}%"), TimeRemainingColumn(),
              console=console) as prog:
    t = prog.add_task("developing video", total=100, completed=100)
console.print("[green]wrote[/] trip.digicam.avi")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cli.svg")
console.save_svg(out, title="digicam2000")
print("wrote", out)
