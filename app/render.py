"""Rendert eine Enten-CSV als Kontaktbogen (PNG) oder Video (MP4)."""
import sys
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion


def camera(qpos_frame, azimuth=140.0):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [qpos_frame[0], qpos_frame[1], 0.085]
    cam.distance, cam.azimuth, cam.elevation = 0.55, azimuth, -14
    return cam


def frames(csv, width, height, step, progress=None):
    fps, qpos, names = motion.load_csv(csv)
    model, data = motion.load_model()
    motion.check_joint_order(model, names)
    look = qpos[0].copy()
    with mujoco.Renderer(model, height, width) as r:
        idx = range(0, len(qpos), step)
        for n, i in enumerate(idx):
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            look[:2] += 0.08 * (qpos[i, :2] - look[:2])
            r.update_scene(data, camera(look))
            if progress:
                progress(n / len(idx))
            yield r.render()


def contact_sheet(csv, out, cols=6, rows=3):
    fps, qpos, _ = motion.load_csv(csv)
    step = max(1, len(qpos) // (cols * rows))
    imgs = list(frames(csv, 480, 400, step))[: cols * rows]
    sheet = Image.new("RGB", (cols * 480, rows * 400))
    for k, im in enumerate(imgs):
        sheet.paste(Image.fromarray(im), ((k % cols) * 480, (k // cols) * 400))
    sheet.save(out)


def video(csv, out, fps_out=30, width=1280, height=720, progress=None):
    import imageio
    step = int(round(motion.CSV_FPS / fps_out))
    with imageio.get_writer(out, fps=fps_out, codec="libx264", quality=8, macro_block_size=None) as w:
        for im in frames(csv, width, height, step, progress):
            w.append_data(im)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    video(src, dst) if dst.endswith(".mp4") else contact_sheet(src, dst)
