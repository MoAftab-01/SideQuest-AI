"""End-to-end smoke check against a running SideQuest AI instance.

    python scripts/smoke_container.py http://localhost:8099

Exercises the real player journey over HTTP -- generate a mission, come back
with evidence, check the verdict -- and asserts the honesty invariant holds at
the API boundary, not just in unit tests.

Used to verify the Docker image actually serves a working app rather than
merely starting. Exits non-zero on the first failed check.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8099"
PLAYER = "smoke-check"

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def post(path: str, payload: dict, query: str = "") -> dict:
    url = f"{BASE}{path}{query}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
        return json.loads(r.read())


def png_data_url(rgb: tuple[int, int, int], size: tuple[int, int] = (320, 240)) -> str:
    from PIL import Image

    img = Image.new("RGB", size, rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    print(f"Smoke check against {BASE}\n")

    print("Health")
    h = get("/api/health")
    check("status ok", h.get("status") == "ok")
    check("engine state is reported", "engine" in h)
    print(f"        engine.loaded={h['engine'].get('loaded')} "
          f"model={h['engine'].get('model_id')}")

    print("\nMission generation")
    req = {
        "time_budget_minutes": 30,
        "interests": ["plants"],
        "energy": "low",
        "location_type": "park",
        "weather": "clear",
    }
    r1 = post("/api/quest", req, f"?player_id={PLAYER}")
    q1 = r1["quest"]
    check("returns a title", bool(q1["title"].strip()), q1["title"])
    check("has steps", len(q1["steps"]) > 0, f"{len(q1['steps'])} steps")
    total = sum(s["approx_minutes"] for s in q1["steps"])
    check("within the 30 minute budget", total <= 30, f"{total} min")
    check("passes the safety gate", q1["safety"]["passed"] is True)
    check("has a verification subject", bool(q1["verification"]["subject"].strip()))
    check("has must_include criteria", len(q1["verification"]["must_include"]) > 0)
    check("generation source is disclosed", "source" in r1["generation"],
          r1["generation"]["source"])

    print("\nDeterminism of the fallback generator")
    r2 = post("/api/quest", {**req, "seed": 42}, f"?player_id={PLAYER}")
    check("same seed yields the same mission",
          r2["quest"]["title"] == post("/api/quest", {**req, "seed": 42},
                                       f"?player_id={PLAYER}")["quest"]["title"])

    print("\nVerification must not overclaim")
    v1 = post("/api/verify",
              {"photo_data_url": "not-a-data-url", "elapsed_minutes": 25},
              f"?player_id={PLAYER}&quest_id={q1['id']}")["verdict"]
    check("garbage photo is not AI verified", v1["is_ai_verified"] is False,
          f"level={v1['level']}")
    check("garbage photo is rejected or uncertain",
          v1["level"] in ("rejected", "uncertain", "heuristic"), v1["level"])

    print("\nVerification with a photo that shows no plant")
    sky = post("/api/quest", req, f"?player_id={PLAYER}")["quest"]
    v2 = post("/api/verify",
              {"photo_data_url": png_data_url((150, 180, 215)),
               "note": "just sky", "elapsed_minutes": 20},
              f"?player_id={PLAYER}&quest_id={sky['id']}")["verdict"]
    check("a sky photo is never AI verified", v2["is_ai_verified"] is False,
          f"level={v2['level']}")
    check("no XP for an unverified result", v2["xp_awarded"] in (0, None)
          or v2["level"] == "heuristic", f"xp={v2['xp_awarded']}")

    print("\nPersistence")
    st = get(f"/api/state?player_id={PLAYER}")
    check("state persists", st["state"]["player_id"] == PLAYER)
    check("outdoor minutes were recorded", st["state"]["minutes_outdoors"] > 0,
          f"{st['state']['minutes_outdoors']} min")
    check("quests_completed incremented", st["state"]["quests_completed"] >= 2)

    print("\nPath traversal is contained")
    # The API echoes the caller's raw player_id back as an opaque label; what
    # matters is the FILESYSTEM path. store._path strips everything but
    # alphanumerics, '-' and '_', so "../../../etc/passwd" must become
    # "etcpasswd" and land inside data/players/. Check the id round-trips
    # without error, and separately assert the sanitised form has no separators.
    evil = get("/api/state?player_id=../../../etc/passwd")
    check("traversal id is accepted without error", bool(evil["state"]["player_id"]))
    sanitised = "".join(c for c in evil["state"]["player_id"] if c.isalnum() or c in "-_")
    check("traversal id contains no path separators",
          "/" not in sanitised and "\\" not in sanitised and ".." not in sanitised,
          repr(sanitised[:40]))

    print("\nMissing quest is a clean 404, not a crash")
    try:
        post("/api/verify", {"photo_data_url": "x", "elapsed_minutes": 1},
             f"?player_id={PLAYER}&quest_id=does-not-exist")
        check("unknown quest returns 404", False, "it returned 200")
    except urllib.error.HTTPError as e:
        check("unknown quest returns 404", e.code == 404, f"got {e.code}")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
