#!/usr/bin/env python3
"""Draw the two paper figures; all geometry and text remain vector objects.

Run from any directory with Python 3 and matplotlib. No model, dataset,
checkpoint, network access, or external LaTeX installation is required.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/handflow-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 180, 151  # millimetres: full-width figure in a two-column paper
COLORS = {
    "ink": "#202B35",
    "line": "#4D5964",
    "muted": "#596572",
    "rule": "#DCE1E6",
    "model": "#E5EDF7",
    "motion": "#E0F0EB",
    "noise": "#EAE5F3",
    "loss": "#F8EEDB",
    "white": "#FFFFFF",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "handflow-pipelines",
    "axes.unicode_minus": False,
})


class Diagram:
    def __init__(self):
        self.fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
        self.ax.set_axis_off()
        self.fig.patch.set_facecolor("white")
        self.contained_text = []

    def text(self, x, y, value, size=8, weight="normal", color="ink", ha="center"):
        return self.ax.text(
            x, y, value, ha=ha, va="center", fontsize=size,
            fontweight=weight, color=COLORS[color], zorder=5,
        )

    def rect(self, x, y, w, h, color="model", outline=True):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.65",
            facecolor=COLORS[color],
            edgecolor=COLORS["line"] if outline else "none",
            linewidth=0.65, zorder=2,
        ))

    def box(self, x, cy, w, h, title, lines=(), color="model", size=8, title_size=8):
        """Text lines are (offset from centre in mm, text, font size)."""
        bottom = cy - h / 2
        self.rect(x, bottom, w, h, color)
        title_y = cy if not lines else cy + h / 2 - 4.1
        title_artist = self.text(x + w / 2, title_y, title, title_size)
        self.contained_text.append((title_artist, (x, bottom, w, h)))
        for offset, value, line_size in lines:
            artist = self.text(x + w / 2, cy + offset, value, line_size or size)
            self.contained_text.append((artist, (x, bottom, w, h)))

    def arrow(self, points, dashed=False, color="line", head=True, width=0.75):
        """Orthogonal polylines prevent ambiguous crossing and curved routing."""
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        if head:
            self.ax.add_patch(FancyArrowPatch(
                path=path, arrowstyle="->", mutation_scale=7,
                linewidth=width, linestyle=(0, (3, 2.5)) if dashed else "solid",
                edgecolor=COLORS[color], facecolor="none", zorder=3,
                capstyle="butt", joinstyle="miter",
            ))
        else:
            self.ax.add_patch(PathPatch(
                path, fill=False, edgecolor=COLORS[color], linewidth=width,
                linestyle=(0, (3, 2.5)) if dashed else "solid", zorder=3,
            ))

    def heading(self, y, letter, title):
        self.text(4, y, f"({letter})  {title}", 9.2, "bold", ha="left")

    def rule(self, y):
        self.ax.plot([4, 177], [y, y], color=COLORS["rule"], lw=0.65, zorder=0)

    def save(self, folder, name, title):
        # Check every box label against its actual rendered extent, including
        # mathematical text, rather than guessing from character counts.
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inverse = self.ax.transData.inverted()
        overflow = []
        for artist, (x, y, w, h) in self.contained_text:
            bounds = artist.get_window_extent(renderer).transformed(inverse)
            if (bounds.x0 < x + 0.65 or bounds.x1 > x + w - 0.65
                    or bounds.y0 < y + 0.5 or bounds.y1 > y + h - 0.5):
                overflow.append(artist.get_text())
        if overflow:
            raise ValueError(f"Labels exceed their boxes: {overflow}")

        destination = ROOT / folder
        destination.mkdir(parents=True, exist_ok=True)
        base = destination / name
        self.fig.savefig(base.with_suffix(".pdf"), metadata={
            "Title": title, "Creator": "HandFlow / matplotlib",
            "CreationDate": None, "ModDate": None,
        })
        self.fig.savefig(base.with_suffix(".svg"), metadata={"Title": title, "Date": None})
        self.fig.savefig(base.with_suffix(".png"), dpi=600, metadata={"Title": title})
        self.fig.savefig(Path("/tmp") / f"{name}_preview.png", dpi=180)
        plt.close(self.fig)
        print(f"Wrote {base}.{{pdf,svg,png}}")


def tiny_unet(d, x, cy):
    """A compact encoder--bottleneck--decoder with a separate skip path."""
    nodes = [(x, cy + 1.4), (x + 7.5, cy - 1.4), (x + 15, cy + 1.4)]
    for bx, by in nodes:
        d.rect(bx, by - 1.6, 5.3, 3.2, "white")
    d.arrow([(x + 5.3, cy + 1.4), (x + 7.5, cy - 1.4)], width=0.6)
    d.arrow([(x + 12.8, cy - 1.4), (x + 15, cy + 1.4)], width=0.6)
    d.arrow([(x + 2.65, cy + 3), (x + 2.65, cy + 5),
             (x + 17.65, cy + 5), (x + 17.65, cy + 3)], width=0.6)


def rectified_flow():
    d = Diagram()
    d.heading(146.5, "a", "Conditional flow matching in motion space")

    # Conditions enter the vector field; sampled t also selects the path point.
    d.box(39, 137, 36, 12, "Flow time", [(-2.4, r"$t\sim\mathcal{U}[0,1]$", 9)])
    d.box(86, 137, 27, 12, "Text", [(-2.4, r"$y$", 10)], color="motion")
    d.box(123, 137, 54, 12, "T5 text encoder (frozen)",
          [(-2.4, r"$c=\mathrm{T5}(y)$", 9)], color="motion")
    d.arrow([(113, 137), (123, 137)])
    d.arrow([(57, 131), (57, 122.5)], dashed=True)
    d.arrow([(75, 137), (81, 137), (81, 123), (86, 123)], dashed=True)
    d.arrow([(150, 131), (150, 129.5), (133, 129.5), (133, 117.3), (129, 117.3)],
            dashed=True)

    d.box(4, 121, 26, 15, "Hand motion", [(-2.4, r"$x_1\in\mathbb{R}^{T\times198}$", 9)],
          color="motion")
    d.box(4, 101, 26, 15, "Gaussian noise", [(-2.4, r"$x_0\sim\mathcal{N}(0,I)$", 9)],
          color="noise", title_size=7.6)
    d.box(39, 111, 38, 23, "Linear interpolation", [
        (0.4, r"$x_t=(1-t)x_0+t x_1$", 9),
        (-6.4, r"$u=x_1-x_0$", 9),
    ], title_size=7.5)
    d.arrow([(30, 121), (34, 121), (34, 116), (39, 116)])
    d.arrow([(30, 101), (34, 101), (34, 106), (39, 106)])

    d.rect(86, 94, 43, 34, "model")
    d.text(107.5, 123, "Temporal 1D U-Net", 8, "bold")
    d.text(107.5, 117.3, "AdaLN or cross-attention", 7.2)
    d.text(107.5, 111.3, r"$v_\theta(x_t,t,c)$", 10)
    tiny_unet(d, 97.3, 100)
    d.arrow([(77, 111), (86, 111)])
    d.text(81.5, 115.5, r"$x_t$", 8.5)

    d.box(141, 111, 36, 24, "Velocity loss", [
        (0.4, r"$\mathcal{L}_{\mathrm{FM}}=$", 9.5),
        (-6.2, r"$\mathbb{E}\,\Vert v_\theta-u\Vert_2^2$", 10),
    ], color="loss")
    d.arrow([(129, 111), (141, 111)])
    d.arrow([(77, 104.6), (81, 104.6), (81, 88.5),
             (135, 88.5), (135, 104.6), (141, 104.6)])
    target_label = d.text(108, 88.5, r"target $u$", 7.6)
    target_label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    d.rule(81)
    d.heading(75, "b", "Rectification (reflow)")
    d.text(177, 75, r"same text condition $c$", 7.6, color="muted", ha="right")
    d.box(4, 56, 26, 18, "Gaussian noise", [(-2.5, r"$x_0\sim\mathcal{N}(0,I)$", 9)],
          color="noise", title_size=7.6)
    d.box(39, 56, 39, 22, "Frozen teacher", [
        (0, r"$\widetilde{x}_1=\mathrm{ODE}(v_\theta,x_0,c)$", 8.6),
        (-6.5, r"$t:0\rightarrow1$", 9),
    ])
    d.box(87, 56, 27, 18, "Coupled pair", [(-2.5, r"$(x_0,\widetilde{x}_1)$", 10)],
          color="motion", title_size=7.6)
    d.box(123, 56, 54, 30, r"Reflow: train $v_\phi$", [
        (3.4, r"$\widetilde{x}_t=(1-t)x_0+t\widetilde{x}_1$", 9),
        (-3.3, r"$\widetilde{u}=\widetilde{x}_1-x_0$", 9),
        (-10, r"$\mathcal{L}_{\mathrm{re}}=\mathbb{E}\,\Vert v_\phi(\widetilde{x}_t,t,c)-\widetilde{u}\Vert_2^2$", 9),
    ], color="loss")
    d.arrow([(30, 56), (39, 56)])
    d.arrow([(78, 56), (87, 56)])
    d.arrow([(114, 56), (123, 56)])

    d.rule(35)
    d.heading(29, "c", "Generation")
    d.box(4, 14, 29, 18, "Gaussian noise", [(-2.5, r"$x_0\sim\mathcal{N}(0,I)$", 9)],
          color="noise")
    d.box(43, 14, 89, 18, "Integrate the rectified field", [
        (-2.5, r"$\mathrm{d}x_t/\mathrm{d}t=v_\phi(x_t,t,c),\quad t:0\rightarrow1$", 10),
    ])
    d.box(142, 14, 35, 18, "Generated motion", [(-2.5, r"$\widehat{x}_1\in\mathbb{R}^{T\times198}$", 9)],
          color="motion", title_size=7.8)
    d.arrow([(33, 14), (43, 14)])
    d.arrow([(132, 14), (142, 14)])
    d.save("hand_flow_base", "rectified_flow_matching", "Rectified flow matching for hand motion")


def latent_flow():
    d = Diagram()
    d.heading(146.5, "a", "Learn the motion VAE")

    d.box(4, 132, 26, 18, "Hand motion", [(-2.5, r"$x\in\mathbb{R}^{T\times198}$", 9)],
          color="motion")
    d.box(38, 132, 28, 18, "VAE encoder", [(-2.5, r"$q_\psi(z\mid x)$", 10)])
    d.box(74, 132, 33, 20, "Sample latent", [
        (0, r"$z\sim q_\psi(z\mid x)$", 9.5),
        (-6.3, "reparameterization", 7),
    ], color="noise")
    d.box(115, 132, 28, 18, "VAE decoder", [(-2.5, r"$D_\omega(z)$", 10)])
    d.box(151, 132, 26, 18, "Reconstruction", [(-2.5, r"$\widehat{x}$", 10)],
          color="motion", title_size=7.3)
    for start, end in [(30, 38), (66, 74), (107, 115), (143, 151)]:
        d.arrow([(start, 132), (end, 132)])

    d.box(38, 108, 105, 18, "Reconstruction + KL regularization", [
        (-2.5, r"$\mathcal{L}_{\mathrm{VAE}}=\mathbb{E}_{q_\psi}\Vert x-D_\omega(z)\Vert_2^2"
               r"+\beta\,D_{\mathrm{KL}}\!\left(q_\psi(z\mid x)\,\Vert\,\mathcal{N}(0,I)\right)$", 9.5),
    ], color="loss")
    d.arrow([(17, 123), (17, 108), (38, 108)])
    d.arrow([(164, 123), (164, 108), (143, 108)])
    d.text(90.5, 96, r"Freeze $q_\psi$ and $D_\omega$ before learning the flow", 7.7,
           color="muted")

    d.rule(91)
    d.heading(85, "b", "Flow matching in latent space")

    d.box(4, 71, 26, 14, "Hand motion", [(-2.2, r"$x$", 10)], color="motion")
    d.box(39, 71, 33, 14, "Frozen encoder", [(-2.2, r"$z_1\sim q_\psi(z\mid x)$", 9)],
          title_size=7.8)
    d.box(4, 51, 26, 14, "Gaussian noise", [(-2.2, r"$z_0\sim\mathcal{N}(0,I)$", 9)],
          color="noise", title_size=7.6)
    d.arrow([(30, 71), (39, 71)])

    d.box(82, 62, 35, 23, "Linear interpolation", [
        (0.2, r"$z_t=(1-t)z_0+t z_1$", 9),
        (-6.2, r"$t\sim\mathcal{U}[0,1]$", 9),
    ], title_size=7.1)
    d.arrow([(72, 71), (77, 71), (77, 67), (82, 67)])
    d.arrow([(30, 51), (76, 51), (76, 57), (82, 57)])

    d.box(127, 62, 50, 23, "Latent velocity model", [
        (0.2, "Temporal 1D U-Net", 7.8),
        (-6.2, r"$v_\theta(z_t,t,c)$", 10),
    ])
    d.arrow([(117, 62), (127, 62)])
    d.text(152, 79.8, r"$t,\ c=\mathrm{T5}(y)$", 8.3)
    d.arrow([(152, 77), (152, 73.5)], dashed=True)
    d.box(82, 39, 95, 12, "Latent velocity loss", [
        (-2.3, r"$\mathcal{L}_{\mathrm{LFM}}=\mathbb{E}\,\Vert v_\theta(z_t,t,c)-(z_1-z_0)\Vert_2^2$", 9.5),
    ], color="loss")
    d.arrow([(152, 50.5), (152, 45)])

    d.rule(29)
    d.heading(23.5, "c", "Generation")
    d.box(4, 10, 26, 14, "Gaussian noise", [(-2.2, r"$z_0\sim\mathcal{N}(0,I)$", 9)],
          color="noise", title_size=7.6)
    d.box(39, 10, 60, 14, "Integrate in latent space", [
        (-2.2, r"$\mathrm{d}z_t/\mathrm{d}t=v_\theta(z_t,t,c),\ t:0\rightarrow1$", 9.1),
    ])
    d.box(108, 10, 32, 14, "Frozen decoder", [(-2.2, r"$D_\omega(\widehat{z}_1)$", 10)], title_size=7.8)
    d.box(149, 10, 28, 14, "Hand motion", [(-2.2, r"$\widehat{x}$", 10)], color="motion")
    for start, end in [(30, 39), (99, 108), (140, 149)]:
        d.arrow([(start, 10), (end, 10)])
    d.save("latent_hand_flow", "latent_flow_matching", "VAE latent flow matching for hand motion")


if __name__ == "__main__":
    rectified_flow()
    latent_flow()
