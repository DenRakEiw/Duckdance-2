"""Haengt die Tanz-Aufgabe in Pollens microduck_rl ein (vendor/microduck_rl).

    python tools/install_training.py

Kopiert training/microduck_dance_env_cfg.py in dessen tasks-Ordner und meldet
die Aufgabe "Mjlab-Dance-Flat-MicroDuck" an. Mehrfaches Ausfuehren ist unschaedlich.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "vendor/microduck_rl/src/mjlab_microduck/tasks"
MARK = "# --- Duck Retarget: dance task ---"
BLOCK = f'''
{MARK}
from . import microduck_dance_env_cfg as _dance
if _dance.MOTION_PATH.exists():
    register_mjlab_task(
        task_id="Mjlab-Dance-Flat-MicroDuck",
        env_cfg=_dance.make_microduck_dance_env_cfg(),
        play_env_cfg=_dance.make_microduck_dance_env_cfg(play=True),
        rl_cfg=_dance.MicroduckDanceRlCfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
'''

if not TASKS.exists():
    raise SystemExit("vendor/microduck_rl fehlt: erst klonen und 'uv sync' ausfuehren (siehe README).")
shutil.copy2(ROOT / "training/microduck_dance_env_cfg.py", TASKS / "microduck_dance_env_cfg.py")
init = TASKS / "__init__.py"
text = init.read_text(encoding="utf-8")
if MARK not in text:
    init.write_text(text.rstrip() + "\n" + BLOCK, encoding="utf-8")
print("Tanz-Aufgabe installiert:", TASKS / "microduck_dance_env_cfg.py")
