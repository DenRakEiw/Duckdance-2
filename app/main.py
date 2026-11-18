"""Duck Retarget: menschliche Bewegung (SOMA-BVH) auf die MicroDuck umrechnen und ansehen.

Start:  DuckRetarget.bat   (oder  .venv\\Scripts\\python.exe app\\main.py)

Die Umrechnung selbst macht der NVIDIA SOMA Retargeter in seiner eigenen
Python-Umgebung (vendor/soma-retargeter); diese App ruft ihn als Unterprozess
auf und zeigt das Ergebnis kinematisch in MuJoCo.
"""
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import mujoco
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion
import render

ROOT = motion.ROOT
VENDOR = ROOT / "vendor" / "soma-retargeter"
SAMPLES = VENDOR / "assets" / "motions" / "bvh"
CLIPS = ROOT / "clips"          # eigene BVH-Dateien
MOTIONS = ROOT / "motions"      # umgerechnete CSV
EXPORT = ROOT / "export"        # Videos und Referenzbewegungen
WORK = ROOT / "work"

BG, CARD, CARD2, LINE, PINK, TEXT, MUTED = "#000000", "#1c1c1c", "#262626", "#343434", "#e8336d", "#ffffff", "#a9a9a9"
SANS, MONO = "Segoe UI", "Consolas"


def clip_files():
    files = sorted(CLIPS.glob("*.bvh")) + sorted(SAMPLES.glob("*.bvh"))
    seen, out = set(), []
    for f in files:
        if f.stem not in seen:
            seen.add(f.stem)
            out.append(f)
    return out


class App(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=BG)
        self.title("Duck Retarget")
        self.geometry("1380x860")
        self.minsize(1100, 700)
        for d in (CLIPS, MOTIONS, EXPORT, WORK):
            d.mkdir(exist_ok=True)

        self.model, self.data = motion.load_model()
        self.renderer = None
        self.view = dict(azimuth=140.0, elevation=-14.0, distance=0.55)
        self.look = np.array([0.0, 0.0])
        self.qpos = None
        self.fps = motion.CSV_FPS
        self.t = 0.0
        self.playing = True
        self.last_tick = time.perf_counter()
        self.selected = None
        self.checked = {}
        self.rows = {}
        self.log_q = queue.Queue()
        self.busy = False

        self._build()
        self.refresh_clips()
        self.after(30, self.tick)
        self.after(100, self.drain_log)

    # ---------------------------------------------------------------- Aufbau
    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(self, fg_color=BG, border_color=PINK, border_width=2, corner_radius=18)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=22, pady=(20, 12))
        ctk.CTkLabel(head, text="HUMAN MOVES. THE DUCK DANCES.", font=(SANS, 26, "bold"), text_color=TEXT).pack(side="left", padx=(26, 30), pady=18)
        ctk.CTkLabel(head, text="SOMA Retargeter  →  MicroDuck", font=(SANS, 14), text_color=MUTED).pack(side="right", padx=26)

        side = ctk.CTkFrame(self, fg_color=CARD, corner_radius=26, width=400)
        side.grid(row=1, column=0, sticky="nsw", padx=(22, 12), pady=(0, 22))
        side.grid_propagate(False)
        side.grid_rowconfigure(2, weight=1)
        side.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(side, text="CLIPS", font=(SANS, 22, "bold")).grid(row=0, column=0, sticky="w", padx=24, pady=(20, 0))
        ctk.CTkLabel(side, text="Motion capture as BVH on the SOMA skeleton.", font=(SANS, 13), text_color=MUTED,
                     ).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 8))
        self.listbox = ctk.CTkScrollableFrame(side, fg_color=CARD, scrollbar_button_color=LINE)
        self.listbox.grid(row=2, column=0, sticky="nsew", padx=12)
        self.listbox.grid_columnconfigure(1, weight=1)

        btns = ctk.CTkFrame(side, fg_color=CARD)
        btns.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 20))
        btns.grid_columnconfigure((0, 1), weight=1)
        self.btn_add = self._ghost(btns, "Add BVH files", self.add_files)
        self.btn_add.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.btn_all = self._ghost(btns, "Select all", self.select_all)
        self.btn_all.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.btn_go = ctk.CTkButton(btns, text="RETARGET TO DUCK", height=50, corner_radius=16, fg_color=PINK,
                                    hover_color="#ff4a84", font=(SANS, 16, "bold"), command=self.retarget)
        self.btn_go.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.progress = ctk.CTkProgressBar(btns, height=4, progress_color=PINK, fg_color=LINE, mode="determinate")
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.progress.set(0)
        self.status = ctk.CTkLabel(btns, text="Ready.", font=(SANS, 13), text_color=MUTED, anchor="w", wraplength=340, justify="left")
        self.status.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        stage = ctk.CTkFrame(self, fg_color=CARD, corner_radius=26)
        stage.grid(row=1, column=1, sticky="nsew", padx=(12, 22), pady=(0, 22))
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_rowconfigure(1, weight=1)
        top = ctk.CTkFrame(stage, fg_color=CARD)
        top.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 8))
        self.title_lbl = ctk.CTkLabel(top, text="STAGE", font=(SANS, 22, "bold"), anchor="w")
        self.title_lbl.pack(side="left")
        self.btn_export = self._ghost(top, "Export motion", self.export_motion)
        self.btn_export.pack(side="right", padx=(8, 0))
        self.btn_video = self._ghost(top, "Save video", self.save_video)
        self.btn_video.pack(side="right", padx=(8, 0))
        self.btn_play = self._ghost(top, "Pause", self.toggle_play)
        self.btn_play.pack(side="right")

        self.screen = tk.Frame(stage, bg="#050505", bd=0)      # feste Flaeche, das Bild darf sie nicht aufziehen
        self.screen.grid(row=1, column=0, sticky="nsew", padx=24)
        self.screen.pack_propagate(False)
        self.canvas = tk.Label(self.screen, bg="#050505", bd=0, cursor="fleur")
        self.canvas.pack(expand=True)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        self.slider = ctk.CTkSlider(stage, from_=0, to=1, progress_color=PINK, button_color=TEXT,
                                    button_hover_color=PINK, fg_color=LINE, command=self.on_seek)
        self.slider.set(0)
        self.slider.grid(row=2, column=0, sticky="ew", padx=24, pady=(12, 0))
        self.stats = ctk.CTkLabel(stage, text="Pick a clip. Drag to orbit, wheel to zoom.", font=(MONO, 13), text_color=MUTED, anchor="w")
        self.stats.grid(row=3, column=0, sticky="ew", padx=24, pady=(6, 4))
        self.log = ctk.CTkTextbox(stage, height=96, fg_color=CARD2, text_color=MUTED, font=(MONO, 11), corner_radius=12)
        self.log.grid(row=4, column=0, sticky="ew", padx=24, pady=(4, 20))
        self.log.configure(state="disabled")

    def _ghost(self, parent, text, cmd):
        return ctk.CTkButton(parent, text=text.upper(), height=34, corner_radius=17, fg_color="transparent", border_width=2,
                             border_color=LINE, hover_color=CARD2, text_color=TEXT, font=(SANS, 12, "bold"), command=cmd)

    # ----------------------------------------------------------------- Clips
    def refresh_clips(self):
        for w in self.listbox.winfo_children():
            w.destroy()
        self.rows = {}
        for r, f in enumerate(clip_files()):
            var = self.checked.setdefault(f.stem, tk.BooleanVar(value=False))
            ctk.CTkCheckBox(self.listbox, text="", variable=var, width=24, checkbox_width=20, checkbox_height=20,
                            fg_color=PINK, hover_color=PINK, border_color=LINE).grid(row=r, column=0, padx=(8, 0), pady=3)
            done = (MOTIONS / f"{f.stem}.csv").exists()
            b = ctk.CTkButton(self.listbox, text=f.stem.replace("_", " "), anchor="w", height=34, corner_radius=10,
                              fg_color=CARD2 if f.stem == self.selected else "transparent", hover_color=CARD2,
                              text_color=TEXT if done else MUTED, font=(SANS, 13),
                              command=lambda s=f.stem: self.show(s))
            b.grid(row=r, column=1, sticky="ew", padx=4, pady=3)
            ctk.CTkLabel(self.listbox, text="●", text_color=PINK if done else LINE, width=18).grid(row=r, column=2, padx=(0, 6))
            self.rows[f.stem] = (f, b)

    def add_files(self):
        paths = filedialog.askopenfilenames(title="Add SOMA BVH files", filetypes=[("BVH motion", "*.bvh")])
        for p in paths:
            if Path(p).resolve() != (CLIPS / Path(p).name).resolve():
                shutil.copy2(p, CLIPS / Path(p).name)
            self.checked.setdefault(Path(p).stem, tk.BooleanVar(value=True)).set(True)
        if paths:
            self.refresh_clips()
            self.say(f"Added {len(paths)} file(s). They must use the SOMA skeleton.")

    def select_all(self):
        new = not all(v.get() for v in self.checked.values())
        for v in self.checked.values():
            v.set(new)

    def show(self, stem):
        self.selected = stem
        csv = MOTIONS / f"{stem}.csv"
        for s, (_, b) in self.rows.items():
            b.configure(fg_color=CARD2 if s == stem else "transparent")
        name = stem.replace("_", " ").upper()
        self.title_lbl.configure(text=name if len(name) <= 36 else name[:35] + "…")
        if not csv.exists():
            self.qpos = None
            self.checked[stem].set(True)
            self.stats.configure(text="Not retargeted yet. Tick it and press RETARGET TO DUCK.")
            return
        try:
            self.fps, self.qpos, names = motion.load_csv(csv)
            motion.check_joint_order(self.model, names)
        except Exception as e:                                   # kaputte oder fremde CSV
            self.qpos = None
            self.stats.configure(text=f"Cannot read {csv.name}: {e}")
            return
        self.t = 0.0
        self.look = self.qpos[0, :2].copy()
        self.playing = True
        self.btn_play.configure(text="PAUSE")
        q = self.qpos[:, 7:]
        lo, hi = self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1]
        at_limit = float(np.mean((q < lo + 0.01) | (q > hi - 0.01))) * 100
        self.stats.configure(text=f"{len(self.qpos) / self.fps:5.1f} s   {len(self.qpos)} frames @ {self.fps:.0f} fps   "
                                  f"joints at limit {at_limit:4.1f} %   root height {self.qpos[:, 2].min() * 100:.1f}"
                                  f" to {self.qpos[:, 2].max() * 100:.1f} cm")

    # ------------------------------------------------------------- Umrechnen
    def retarget(self):
        if self.busy:
            return
        stems = [s for s, v in self.checked.items() if v.get() and s in self.rows]
        if not stems:
            self.say("Tick at least one clip first.")
            return
        uv = shutil.which("uv")
        if not uv or not (VENDOR / ".venv").exists():
            self.say("SOMA Retargeter environment not found. Run 'uv sync' in vendor/soma-retargeter.")
            return
        job = WORK / "job_in"
        shutil.rmtree(job, ignore_errors=True)
        job.mkdir(parents=True)
        for s in stems:
            shutil.copy2(self.rows[s][0], job / f"{s}.bvh")
        cfg = WORK / "job.json"
        cfg.write_text(json.dumps({
            "import_folder": str(job), "export_folder": str(MOTIONS), "batch_size": 10,
            "retargeter": "Newton", "retarget_source": "soma", "retarget_target": "microduck",
            "retarget_source_facing_direction": "Mujoco", "extra_robot_paths": [str(ROOT / "robots")],
        }, indent=2), encoding="utf-8")
        self.busy = True
        self.btn_go.configure(state="disabled", text="RETARGETING …")
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        self.say(f"Retargeting {len(stems)} clip(s) on the GPU. The first run compiles kernels and takes a few minutes.")
        cmd = [uv, "run", "python", "app/bvh_to_csv_converter.py", "--viewer", "null", "--config", str(cfg)]
        threading.Thread(target=self._run_job, args=(cmd, stems), daemon=True).start()

    def _run_job(self, cmd, stems):
        t0 = time.time()
        try:
            p = subprocess.Popen(cmd, cwd=VENDOR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in p.stdout:
                line = line.rstrip()
                if line and "it/s]" not in line and "s/it]" not in line:
                    self.log_q.put(("log", line))
            code = p.wait()
        except Exception as e:
            self.log_q.put(("log", f"{type(e).__name__}: {e}"))
            code = -1
        made = [s for s in stems if (MOTIONS / f"{s}.csv").exists() and (MOTIONS / f"{s}.csv").stat().st_mtime >= t0]
        self.log_q.put(("done", (code, made, stems, time.time() - t0)))

    def drain_log(self):
        try:
            while True:
                kind, payload = self.log_q.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "say":
                    self.say(payload)
                elif kind == "idle":
                    self.busy = False
                    self.progress.stop()
                    self.progress.configure(mode="determinate")
                    self.progress.set(0)
                else:
                    code, made, stems, secs = payload
                    self.busy = False
                    self.progress.stop()
                    self.progress.configure(mode="determinate")
                    self.progress.set(0)
                    self.btn_go.configure(state="normal", text="RETARGET TO DUCK")
                    self.refresh_clips()
                    if code == 0 and len(made) == len(stems):
                        self.say(f"Done: {len(made)} clip(s) in {secs:.0f} s.")
                    else:
                        self.say(f"Failed (exit code {code}): {len(made)} of {len(stems)} clip(s) written. See the log below.")
                    if made:
                        self.show(made[0])
        except queue.Empty:
            pass
        self.after(100, self.drain_log)

    def say(self, text):
        self.status.configure(text=text)

    # --------------------------------------------------------------- Ausgabe
    def save_video(self):
        if self.qpos is None or self.busy:
            return self.say("Pick a retargeted clip first.")
        src, out = MOTIONS / f"{self.selected}.csv", EXPORT / f"{self.selected}.mp4"
        self.busy = True
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        self.say("Rendering video …")

        def work():
            try:
                render.video(src, out)
                self.log_q.put(("say", f"Video saved: {out}"))
            except Exception as e:
                self.log_q.put(("say", f"Video failed: {type(e).__name__}: {e}"))
            self.log_q.put(("idle", None))
        # eigener Renderer im Arbeits-Thread, die Vorschau laeuft im Hauptthread weiter
        threading.Thread(target=work, daemon=True).start()

    def export_motion(self):
        """Referenzbewegung fuer das Training einer Tracking-Policy: 50 Hz, MuJoCo-Layout."""
        if self.qpos is None:
            return self.say("Pick a retargeted clip first.")
        hz = 50.0
        idx = np.round(np.arange(0, len(self.qpos) / self.fps, 1 / hz) * self.fps).astype(int)
        idx = idx[idx < len(self.qpos)]
        q = self.qpos[idx]
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, self.model.njnt)]
        out = EXPORT / f"{self.selected}.npz"
        np.savez_compressed(out, fps=hz, joint_names=np.array(names), root_pos=q[:, 0:3], root_quat_wxyz=q[:, 3:7],
                            joint_pos=q[:, 7:], qpos=q)
        (EXPORT / f"{self.selected}.json").write_text(json.dumps({
            "fps": hz, "joint_names": names, "frames": len(q), "qpos": np.round(q, 5).tolist()}), encoding="utf-8")
        self.say(f"Motion exported at 50 Hz: {out} (+ .json)")

    # --------------------------------------------------------------- Vorschau
    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.configure(text="PAUSE" if self.playing else "PLAY")

    def on_seek(self, v):
        if self.qpos is not None:
            self.t = float(v) * len(self.qpos) / self.fps

    def on_press(self, e):
        self._drag = (e.x, e.y)

    def on_drag(self, e):
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self.view["azimuth"] = (self.view["azimuth"] - dx * 0.4) % 360
        self.view["elevation"] = float(np.clip(self.view["elevation"] - dy * 0.3, -89, 10))

    def on_wheel(self, e):
        self.view["distance"] = float(np.clip(self.view["distance"] * (0.9 if e.delta > 0 else 1.1), 0.18, 3.0))

    def tick(self):
        now = time.perf_counter()
        dt, self.last_tick = now - self.last_tick, now
        w = min(1920, max(320, self.screen.winfo_width())) // 2 * 2
        h = min(1080, max(240, self.screen.winfo_height())) // 2 * 2
        if self.renderer is None or (self.renderer.width, self.renderer.height) != (w, h):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, h, w)
        if self.qpos is not None:
            dur = len(self.qpos) / self.fps
            if self.playing:
                self.t = (self.t + dt) % dur
                self.slider.set(self.t / dur)
            frame = self.qpos[min(len(self.qpos) - 1, int(self.t * self.fps))]
            self.data.qpos[:] = frame
            self.look += min(1.0, 4 * dt) * (frame[:2] - self.look)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.look[:] = 0
        mujoco.mj_forward(self.model, self.data)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [self.look[0], self.look[1], 0.085]
        cam.distance, cam.azimuth, cam.elevation = self.view["distance"], self.view["azimuth"], self.view["elevation"]
        self.renderer.update_scene(self.data, cam)
        self._img = ImageTk.PhotoImage(Image.fromarray(self.renderer.render()))
        self.canvas.configure(image=self._img)
        self.after(30, self.tick)


def _shot(app, path, stem):
    """Nur zum Testen: Clip waehlen, Fenster abfotografieren, beenden."""
    from PIL import ImageGrab
    if stem:
        app.show(stem)
    def grab():
        x, y = app.winfo_rootx(), app.winfo_rooty()
        ImageGrab.grab((x, y, x + app.winfo_width(), y + app.winfo_height())).save(path)
        app.destroy()
    app.after(2500, grab)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    app = App()
    if "--shot" in sys.argv:
        i = sys.argv.index("--shot")
        app.attributes("-topmost", True)
        app.after(800, lambda: _shot(app, sys.argv[i + 1], sys.argv[i + 2] if len(sys.argv) > i + 2 else None))
    app.mainloop()
