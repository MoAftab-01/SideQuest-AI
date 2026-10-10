"""Generate the labelled verification case set.

    python eval/make_cases.py

Why these images are drawn rather than photographed
---------------------------------------------------
A credible eval needs a labelled set of images that exercise the verifier's
decisions. Fetching photographs would mean bundling third-party media and
attributing it, and photographs of a *specific* mission subject are not
publicly available under a licence that suits a demo repo.

So the cases are drawn with Pillow, and this module says so loudly rather than
letting the results imply otherwise. What that means for the numbers:

  * `false_verified_count` IS meaningful on these images. It measures whether
    the system over-claims verification, which depends on our prompt, parsing,
    refusal handling and the `is_ai_verified` invariant -- not on the pixels.
    A model that fails the honesty invariant fails it here.
  * `accuracy_on_decidable_pct` is NOT a claim about photographic perception.
    Drawn shapes with gradients, noise and depth-of-field blur are out of
    distribution for an image encoder. Read it as "the pipeline reaches a
    defensible decision", not "the model recognises flowers as well as a
    person does".

Swapping in real photographs later is a matter of dropping JPEGs into
`eval/data/images/` and pointing the manifest at them; nothing else changes.

The set is aimed at the mission produced by `fallback_quest(QuestRequest(
time_budget_minutes=30, interests=["plants"], seed=42))`, whose spec is:

    subject      a close photograph of a seasonal growth stage: a bud, an open
                 flower, a fruit or a seed head, or bare twigs.
    must_include a plant is the main subject; the growth stage is visible; outdoors
    reject_if    houseplants; cut flowers in a vase; a tree with no seasonal detail
    ambiguous    growth stage cannot be read; the image is too blurred
"""

from __future__ import annotations

import base64
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE / "data" / "images"
MANIFEST = HERE / "data" / "labeled_cases.json"

W, H = 768, 576


# ---------------------------------------------------------------------------
# Drawing helpers
#
# Each case is built from the same vocabulary: a sky-to-ground gradient, some
# out-of-focus background blobs, a subject, then grain and a shallow-focus
# blur. The point is not photorealism -- it is that no case is a flat cartoon
# on white, so the encoder sees something in the shape of a photo.
# ---------------------------------------------------------------------------


def _sky(seed: int, hue: tuple[int, int, int]) -> Image.Image:
    rnd = random.Random(seed)
    top = tuple(max(0, c - rnd.randint(10, 45)) for c in hue)
    bottom = tuple(min(255, c + rnd.randint(10, 50)) for c in hue)

    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        t = y / (H - 1)
        # The `* t` belongs on the delta, not on the whole sum. Wrapping the sum
        # instead makes this `bottom[i] * t`: pure black at the top of the frame
        # fading to full daylight at the horizon. Every outdoor case then read as
        # dusk, which directly contradicted the mission's "outdoors" requirement.
        row = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(W):
            px[x, y] = row
    return img


def _bokeh(img: Image.Image, seed: int, n: int = 26, radius: int = 34) -> Image.Image:
    """Soft out-of-focus highlights, the way a real lens renders background."""
    rnd = random.Random(seed + 991)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(n):
        x, y = rnd.randint(-40, W + 40), rnd.randint(-40, H + 40)
        r = rnd.randint(radius // 2, radius)
        g = rnd.randint(120, 245)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(g, g, rnd.randint(90, 200), rnd.randint(40, 110)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius // 2))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def _grain(img: Image.Image, seed: int, amount: int = 9) -> Image.Image:
    """Sensor noise. Keeps the image off the clean-vector manifold."""
    rnd = random.Random(seed + 4242)
    px = img.load()
    for y in range(H):
        for x in range(W):
            n = rnd.randint(-amount, amount)
            r, g, b = px[x, y]
            px[x, y] = (
                max(0, min(255, r + n)),
                max(0, min(255, g + n)),
                max(0, min(255, b + n)),
            )
    return img


def _vignette(img: Image.Image) -> Image.Image:
    """Corner falloff, kept gentle.

    An early version darkened the top third into near-black. That made outdoor
    cases read as night-time and fight the mission's "outdoors" requirement. A
    vignette should shade corners, not relight the scene -- so it stays mild.

    (The genuinely near-black sky in outdoor cases was a separate bug in `_sky`,
    not this function. This one only ever cost the corners some brightness.)
    """
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([-W // 3, -H // 3, W + W // 3, H + H // 3], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(120))
    # 210 rather than 255: corners darken, the centre stays full brightness.
    mask = mask.point(lambda v: 150 + int(105 * v / 255))
    dark = Image.new("RGB", (W, H), (0, 0, 0))
    return Image.composite(img, dark, mask)


def _finish(img: Image.Image, seed: int, blur: float = 1.2) -> Image.Image:
    img = img.filter(ImageFilter.GaussianBlur(blur))
    img = _vignette(_grain(img, seed))
    return img


def _petal(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, angle: float, color: tuple[int, int, int]) -> None:
    """One petal as a filled ellipse rotated about the flower centre."""
    ca, sa = math.cos(angle), math.sin(angle)
    points = []
    for i in range(24):
        t = 2 * math.pi * i / 24
        # Teardrop: long on one axis, so the petal reads as a petal.
        px_, py_ = r * math.cos(t), r * 1.7 * math.sin(t) * (0.45 + 0.55 * (1 + math.cos(t)) / 2)
        points.append((cx + px_ * ca - py_ * sa, cy + px_ * sa + py_ * ca))
    d.polygon(points, fill=color)


def _open_flower(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, petal_rgb: tuple[int, int, int]) -> None:
    for i in range(8):
        _petal(d, cx, cy, r, 2 * math.pi * i / 8, petal_rgb)
    d.ellipse([cx - r * 0.42, cy - r * 0.42, cx + r * 0.42, cy + r * 0.42], fill=(196, 150, 44))


def _stem(d: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, width: int = 13) -> None:
    d.line([(x0, y0), (x1, y1)], fill=(58, 104, 46), width=width)


def _leaf(d: ImageDraw.ImageDraw, cx: float, cy: float, half_w: float, angle: float, color: tuple[int, int, int]) -> None:
    """A leaf as a rotated ellipse.

    A circle reads as a berry. An ellipse twice as long as it is wide, pointing
    along the petiole, reads as a leaf.
    """
    ca, sa = math.cos(angle), math.sin(angle)
    points = []
    for i in range(28):
        t = 2 * math.pi * i / 28
        px_, py_ = half_w * math.cos(t), half_w * 2.0 * math.sin(t)
        points.append((cx + px_ * ca - py_ * sa, cy + px_ * sa + py_ * ca))
    d.polygon(points, fill=color)


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------


def case_open_flower() -> Image.Image:
    """An open flower, close, outdoors, growth stage unmistakable."""
    img = _sky(1, (128, 176, 226))
    img = _bokeh(img, 1, n=30)
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(H * 0.74), W, H], fill=(74, 116, 58))  # ground
    _stem(d, 300, H, 372, 250, 17)
    d.ellipse([300, 300, 420, 380], fill=(64, 116, 52))  # a leaf
    _open_flower(d, 380, 226, 82, (226, 96, 122))
    return _finish(img, 1, 1.0)


def case_seed_head() -> Image.Image:
    """A seed head against the sky -- a valid alternative growth stage.

    The first version of this drew two columns of small beige ellipses on a
    15px stem, which measured at 0.8% of the frame and read on inspection as
    dead twigs -- the opposite of what the label claims. A seed head is legible
    because of its *silhouette*: a sphere of fine radiating filaments. That is
    what gets drawn here, at a size where the shape, not the pixel count, is
    what the encoder sees.
    """
    img = _sky(2, (150, 186, 224))
    img = _bokeh(img, 2, n=24)
    d = ImageDraw.Draw(img)

    cx, cy, r = 372, 208, 118
    _stem(d, 344, H, cx + 6, cy + r - 6, 20)  # the stem stays green: this is a living plant

    rnd = random.Random(77)
    # Radiating filaments. Two passes -- a dense under-layer at full length, then
    # a sparser over-layer -- so the crown has depth instead of reading as a
    # wireframe ball.
    for a in range(230):
        ang = 2 * math.pi * a / 230 + rnd.uniform(-0.03, 0.03)
        length = r * rnd.uniform(0.74, 1.0)
        x2, y2 = cx + math.cos(ang) * length, cy + math.sin(ang) * length
        d.line([(cx, cy), (x2, y2)], fill=(206, 196, 168), width=2)
    for a in range(130):
        ang = 2 * math.pi * a / 130 + 0.11
        length = r * rnd.uniform(0.6, 0.92)
        x2, y2 = cx + math.cos(ang) * length, cy + math.sin(ang) * length
        d.line([(cx, cy), (x2, y2)], fill=(226, 216, 190), width=2)
        # A seed tuft: the little parachute that makes it a seed head, not a ball.
        for k in range(4):
            ta = ang + (k - 1.5) * 0.30
            d.line(
                [(x2, y2), (x2 + math.cos(ta) * 15, y2 + math.sin(ta) * 15)],
                fill=(238, 230, 206),
                width=2,
            )

    d.ellipse([cx - 20, cy + r * 0.62, cx + 20, cy + r * 0.95], fill=(126, 138, 74))  # receptacle
    return _finish(img, 2, 1.1)


def case_houseplant() -> Image.Image:
    """A plant, but a houseplant: `reject_if` names this explicitly."""
    img = Image.new("RGB", (W, H), (222, 218, 210))  # indoor wall
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(H * 0.68), W, H], fill=(178, 168, 156))  # floor
    d.rounded_rectangle([292, 430, 476, 600], 16, fill=(196, 122, 84))  # pot
    d.rounded_rectangle([302, 412, 466, 452], 10, fill=(178, 106, 72))
    rnd = random.Random(9)
    # Each leaf gets its own origin and its own length. An earlier version
    # fanned 11 leaves from one shared point at a narrow angle spread, which
    # merged them into a single green wedge -- nothing in a real home looks like
    # that, so the case was really testing the drawing code, not the model.
    for i in range(7):
        ang = -math.pi / 2 + (i - 3) * 0.40 + rnd.uniform(-0.07, 0.07)
        length = rnd.uniform(120, 210)
        ox = 384 + rnd.uniform(-18, 18)
        oy = 424 + rnd.uniform(-8, 8)
        x2 = ox + math.cos(ang) * length
        y2 = oy + math.sin(ang) * length
        d.line([(ox, oy), (x2, y2)], fill=(46, 92, 40), width=7)
        _leaf(d, x2, y2, rnd.uniform(24, 36), ang, (64, 126, 52))
    for _ in range(3):  # a window's light, indoors
        x = rnd.randint(80, 600)
        d.rectangle([x, 40, x + 90, 190], fill=(246, 244, 236))
    return _finish(img, 3, 1.3)


def case_cut_flowers() -> Image.Image:
    """A cut flower in a vase: present, healthy, and explicitly rejected."""
    img = Image.new("RGB", (W, H), (206, 202, 198))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(H * 0.72), W, H], fill=(150, 142, 134))
    d.rounded_rectangle([330, 430, 440, 600], 12, fill=(212, 214, 218))  # glass vase
    for i, dx in enumerate((-34, 0, 34)):
        _stem(d, 385, 440, 385 + dx, 250, 8)
        _open_flower(d, 385 + dx, 226 - abs(dx) // 2, 46, ((228, 122, 96), (232, 176, 88), (206, 118, 168))[i])
    return _finish(img, 4, 1.2)


def case_blurred() -> Image.Image:
    """A plant is plainly there but unreadable: the honest answer is `ambiguous`."""
    img = case_open_flower()
    # 48 rather than 26: at 26 the flower head was still legible, which would
    # have made this an accuracy case dressed up as an ambiguity case.
    return img.filter(ImageFilter.GaussianBlur(48))


def case_distant_tree() -> Image.Image:
    """A tree with no identifiable seasonal detail: `reject_if` names this."""
    img = _sky(6, (140, 178, 214))
    img = _bokeh(img, 6, n=18)
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(H * 0.80), W, H], fill=(86, 122, 66))
    d.rectangle([356, 250, 412, H], fill=(78, 62, 48))
    rnd = random.Random(31)
    for _ in range(190):  # canopy mass, no structure inside it
        x, y = rnd.gauss(384, 118), rnd.gauss(206, 78)
        r = rnd.randint(20, 46)
        g = rnd.randint(74, 128)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(g, int(g * 1.35), int(g * 0.62)))
    return _finish(img, 6, 1.4)


def case_sky_only() -> Image.Image:
    """No plant at all."""
    img = _sky(7, (110, 158, 214))
    img = _bokeh(img, 7, n=12)
    return _finish(img, 7, 1.0)


def case_bare_twigs() -> Image.Image:
    """Bare winter twigs against sky: a listed acceptable variation.

    The first version compounded a +/-0.55 rad random turn over five levels from
    a 150px trunk, so the crown collapsed into a ~1.7%-of-frame tangle of 2px
    lines that no longer read as a plant at all. A second attempt fixed the
    thickness but raised the trunk to 300px without rechecking the budget: with
    0.69 decay the total path was 300 + 228 + 157 + 108 + 75 ~= 868px of drawing
    into a 576px frame, so the trunk ran straight off the top edge and every
    branch past level 1 was outside the image.

    So the length is now derived from the frame rather than guessed, and the
    branching alternates sign instead of jittering: with `ang + uniform(-0.3,
    0.3)` on both children the whole crown stayed within ~25 degrees of the
    trunk axis and collapsed into one clump, which read as a flowering stalk
    rather than a bare shrub. Children now leave the parent at a *wider* angle
    than the parent did, which is what makes a candelabra.

    Budget: with 0.75 decay the total path is L * 3.05, so L = 150 draws ~458px
    into a 576px frame and the crown lands with sky around it.
    """
    img = _sky(8, (168, 192, 218))
    img = _bokeh(img, 8, n=16)
    d = ImageDraw.Draw(img)
    rnd = random.Random(53)
    DEPTH = 4

    def branch(x: float, y: float, ang: float, length: float, depth: int, width: int) -> None:
        x2, y2 = x + math.cos(ang) * length, y + math.sin(ang) * length
        d.line([(x, y), (x2, y2)], fill=(84, 68, 58), width=width)
        if depth == 0:
            # A terminal bud -- the one seasonal detail that makes this decidable.
            d.ellipse([x2 - 9, y2 - 13, x2 + 9, y2 + 13], fill=(158, 112, 92))
            return
        # Wider than the parent branch left at, and the two children diverge to
        # opposite sides. Jitter is small enough not to undo the splay.
        spread = 0.30 + 0.09 * (DEPTH - depth)
        for sign in (-1, 1):
            branch(
                x2, y2,
                ang + sign * spread + rnd.uniform(-0.08, 0.08),
                length * rnd.uniform(0.70, 0.80),
                depth - 1,
                max(5, width - 2),
            )

    branch(360, H, -math.pi / 2 + 0.06, 150, DEPTH, 17)
    return _finish(img, 8, 1.1)


def case_no_photo_reference() -> Image.Image:
    """Placeholder body for the "note only, no photo" path (never sent to the model)."""
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    return img


#: (filename, builder, expected, note)
#:
#: `expected` values the harness scores on:
#:   verified  - the image plainly satisfies the mission spec
#:   rejected  - the image plainly fails it, for a reason in `reject_if`
#:   ambiguous - genuinely undecidable; held OUT of accuracy, reported separately
CASES: list[tuple[str, Any, str, str]] = [
    ("open_flower.jpg", case_open_flower, "verified",
     "Open flower, close, outdoors. Growth stage unmistakable."),
    ("seed_head.jpg", case_seed_head, "verified",
     "Seed head against sky -- a listed alternative growth stage."),
    ("bare_twigs.jpg", case_bare_twigs, "verified",
     "Bare winter twigs with terminal buds; an accept_hint."),
    ("houseplant.jpg", case_houseplant, "rejected",
     "Healthy plant, wrong context: houseplants are in reject_if."),
    ("cut_flowers.jpg", case_cut_flowers, "rejected",
     "Cut flowers in a vase; also in reject_if."),
    ("distant_tree.jpg", case_distant_tree, "rejected",
     "A tree whose canopy hides all seasonal detail; in reject_if."),
    ("sky_only.jpg", case_sky_only, "rejected",
     "No plant anywhere in frame."),
    ("blurred_plant.jpg", case_blurred, "ambiguous",
     "Plant present but unreadable. Held out of accuracy by design."),
]


def build(*, write_images: bool = True) -> dict[str, Any]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []

    for name, builder, expected, note in CASES:
        path = IMAGE_DIR / name
        if write_images:
            builder().save(path, "JPEG", quality=86)
        data_url = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        cases.append({
            "id": path.stem,
            "file": f"eval/data/images/{name}",
            "photo_data_url": data_url,
            "expected": expected,
            "note": note,
            "elapsed_minutes": 20,
        })

    manifest = {
        "generated_by": "eval/make_cases.py",
        "synthetic": True,
        "synthetic_note": (
            "These images are drawn with Pillow, not photographed. false_verified_count "
            "is meaningful on them (it measures our honesty invariant, not perception). "
            "accuracy_on_decidable_pct is NOT a claim about photographic recognition -- "
            "drawn shapes are out of distribution for an image encoder. See the module "
            "docstring in eval/make_cases.py."
        ),
        "mission": (
            "Authored against fallback_quest(QuestRequest(time_budget_minutes=30, "
            "interests=['plants'], seed=42)) -- the same quest the harness builds by default."
        ),
        "cases": cases,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build()
    counts: dict[str, int] = {}
    for c in m["cases"]:
        counts[c["expected"]] = counts.get(c["expected"], 0) + 1
    print(f"Wrote {len(m['cases'])} cases to {MANIFEST}")
    print(f"  scored:   {counts.get('verified', 0)} verified + {counts.get('rejected', 0)} rejected"
          f" = {counts.get('verified', 0) + counts.get('rejected', 0)} decidable")
    print(f"  held out: {counts.get('ambiguous', 0)} ambiguous")
    print(f"  images:   {IMAGE_DIR}")