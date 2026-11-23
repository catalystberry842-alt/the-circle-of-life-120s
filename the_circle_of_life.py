#!/usr/bin/env python3
"""
"The Circle of Life" — 120-Second Procedural Cinematic Masterpiece
===================================================================

Pure Python Graphics & Audio DSP Pipeline (`moviepy`, `numpy`, `cv2`, `scipy`).
Zero external images, textures, meshes, or audio files.

Duration: Exactly 120.0 seconds (2,880 frames @ 24 FPS, 1280x720 HD)
Output:   the_circle_of_life_120s.mp4

Architecture (5 Strict Modules):
    1. SDF_Renderer    : Vectorized 2D SDFs, smooth-min blending, 2.5D normals,
                         fake Subsurface Scattering (SSS), view-angle Iridescence,
                         animated Voronoi camouflage, directional feather noise.
    2. Lighting_Engine : Continuous `time_of_day` global illumination, weather,
                         volumetric god rays, parallax jungle, Thermal Heatmap filter.
    3. Particle_System : Deterministic vectorized physics for rain, expanding
                         water ripples, glowing spores, golden pollen & 3D lens feathers.
    4. DSP_Audio_Synth : Brian Eno-style generative ambient drone (5 detuned LFO
                         voices + Schroeder algorithmic reverb) & binaural X-panned foley.
    5. Cinematographer : 7-Act choreography, L-system mycelium, Fibonacci unrolling
                         fractal fern, 2-bone analytical IK, camera pan/shake &
                         parallel multiprocessing chunk renderer -> MoviePy MP4.
"""

from __future__ import annotations

import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    from moviepy.editor import AudioFileClip, VideoClip
except ImportError:  # MoviePy 2.x compatibility fallback
    from moviepy import AudioFileClip, VideoClip


# =============================================================================
# Global Constants
# =============================================================================

WIDTH = 1280
HEIGHT = 720
FPS = 24
DURATION = 120.0
TOTAL_FRAMES = int(round(DURATION * FPS))  # 2880 frames
SAMPLE_RATE = 44100
CHUNK_SIZE = 16  # Frames per parallel multiprocessing worker chunk


# =============================================================================
# Math & Easing Utilities
# =============================================================================


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def lerp_vec(a: Sequence[float], b: Sequence[float], u: float) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) * (1.0 - u) + np.asarray(b, dtype=np.float32) * u


def smoothstep(u: float) -> float:
    u = clamp(u)
    return u * u * (3.0 - 2.0 * u)


def smoothstep_arr(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_in_quad(u: float) -> float:
    u = clamp(u)
    return u * u


def ease_out_quad(u: float) -> float:
    u = clamp(u)
    return 1.0 - (1.0 - u) * (1.0 - u)


def ease_in_cubic(u: float) -> float:
    u = clamp(u)
    return u * u * u


def ease_out_cubic(u: float) -> float:
    u = clamp(u)
    return 1.0 - (1.0 - u) ** 3


def ease_in_out_cubic(u: float) -> float:
    u = clamp(u)
    return 4.0 * u * u * u if u < 0.5 else 1.0 - ((-2.0 * u + 2.0) ** 3) * 0.5


def solve_ik_2bone(
    hip: Tuple[float, float],
    target: Tuple[float, float],
    l1: float,
    l2: float,
    bend_sign: float = 1.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Analytical 2-link cosine-law Inverse Kinematics solver returning (knee, foot)."""
    hx, hy = hip
    tx, ty = target
    dx, dy = tx - hx, ty - hy
    dist = math.hypot(dx, dy)
    max_r = max(1e-3, (l1 + l2) * 0.985)
    min_r = abs(l1 - l2) + 1e-3
    dist_c = clamp(dist, min_r, max_r)
    base_ang = math.atan2(dy, dx)
    cos_a = clamp((l1 * l1 + dist_c * dist_c - l2 * l2) / (2.0 * l1 * dist_c), -1.0, 1.0)
    ang = base_ang + bend_sign * math.acos(cos_a)
    kx = hx + l1 * math.cos(ang)
    ky = hy + l1 * math.sin(ang)
    # Recompute exact reachable foot position
    if dist > max_r:
        fx = hx + dx * (max_r / dist)
        fy = hy + dy * (max_r / dist)
    else:
        fx, fy = tx, ty
    return (kx, ky), (fx, fy)


def generate_lsystem(axiom: str, rules: Dict[str, str], iterations: int) -> str:
    s = axiom
    for _ in range(iterations):
        s = "".join(rules.get(ch, ch) for ch in s)
    return s


# =============================================================================
# MODULE 1: SDF_Renderer (2D SDFs, smin, SSS, Iridescence, Voronoi, Plumage)
# =============================================================================


class SDF_Renderer:
    """
    Vectorized float32 2D Signed Distance Field engine with smooth-min organic
    blending, pseudo-3D normal lifting, fake Subsurface Scattering (SSS),
    view-angle Iridescence, animated Voronoi camouflage, and directional
    high-frequency feather noise shading.
    """

    @staticmethod
    def make_grid(
        x0: int, y0: int, x1: int, y1: int, scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        w = max(4, int(round((x1 - x0) * scale)))
        h = max(4, int(round((y1 - y0) * scale)))
        xs = np.linspace(x0, x1 - 1, w, dtype=np.float32)
        ys = np.linspace(y0, y1 - 1, h, dtype=np.float32)
        return np.meshgrid(xs, ys)

    @staticmethod
    def smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
        """Polynomial smooth minimum for seamlessly melting organic SDF limbs."""
        if k <= 1e-5:
            return np.minimum(a, b)
        h = np.maximum(k - np.abs(a - b), np.float32(0.0)) / np.float32(k)
        return np.minimum(a, b) - h * h * np.float32(k * 0.25)

    @staticmethod
    def smax(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
        """Polynomial smooth maximum for smooth boolean subtraction/carving."""
        return -SDF_Renderer.smin(-a, -b, k)

    @staticmethod
    def sd_circle(
        X: np.ndarray, Y: np.ndarray, cx: float, cy: float, r: float
    ) -> np.ndarray:
        return np.hypot(X - np.float32(cx), Y - np.float32(cy)) - np.float32(r)

    @staticmethod
    def sd_ellipse(
        X: np.ndarray,
        Y: np.ndarray,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        angle_rad: float = 0.0,
    ) -> np.ndarray:
        """Approximate signed distance to an oriented 2D ellipse (fast & smooth)."""
        dx = X - np.float32(cx)
        dy = Y - np.float32(cy)
        if abs(angle_rad) > 1e-5:
            c = np.float32(math.cos(-angle_rad))
            s = np.float32(math.sin(-angle_rad))
            px = dx * c - dy * s
            py = dx * s + dy * c
        else:
            px, py = dx, dy
        inv_rx = np.float32(1.0 / max(1e-3, rx))
        inv_ry = np.float32(1.0 / max(1e-3, ry))
        k0 = np.hypot(px * inv_rx, py * inv_ry)
        k1 = np.hypot(px * inv_rx * inv_rx, py * inv_ry * inv_ry)
        return np.where(k1 > 1e-6, k0 * (k0 - np.float32(1.0)) / k1, -min(rx, ry))

    @staticmethod
    def sd_capsule(
        X: np.ndarray,
        Y: np.ndarray,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        r1: float,
        r2: Optional[float] = None,
    ) -> np.ndarray:
        """Tapered 2D capsule SDF from (ax, ay) [radius r1] to (bx, by) [radius r2]."""
        pax = X - np.float32(ax)
        pay = Y - np.float32(ay)
        bax = np.float32(bx - ax)
        bay = np.float32(by - ay)
        denom = bax * bax + bay * bay + np.float32(1e-6)
        h = np.clip((pax * bax + pay * bay) / denom, np.float32(0.0), np.float32(1.0))
        d = np.hypot(pax - bax * h, pay - bay * h)
        if r2 is None or abs(r2 - r1) < 1e-4:
            return d - np.float32(r1)
        return d - (np.float32(r1) + np.float32(r2 - r1) * h)

    @staticmethod
    def sd_polyline_tapered(
        X: np.ndarray,
        Y: np.ndarray,
        pts: Sequence[Tuple[float, float]],
        radii: Sequence[float],
        blend_k: float = 4.0,
    ) -> np.ndarray:
        """Smoothly blend a chain of tapered capsules along a curved spline."""
        d = SDF_Renderer.sd_capsule(
            X, Y, pts[0][0], pts[0][1], pts[1][0], pts[1][1], radii[0], radii[1]
        )
        for i in range(1, len(pts) - 1):
            seg = SDF_Renderer.sd_capsule(
                X, Y, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], radii[i], radii[i + 1]
            )
            d = SDF_Renderer.smin(d, seg, blend_k)
        return d

    @staticmethod
    def compute_normals_3d(
        d: np.ndarray, max_depth: float = 34.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Lift a 2D SDF field into a rounded 3D surface normal (Nx, Ny, Nz) and
        normalized interior dome height `depth in [0, 1]`.
        """
        gy, gx = np.gradient(d)
        glen = np.hypot(gx, gy) + np.float32(1e-5)
        nx2 = gx / glen
        ny2 = gy / glen

        depth = np.clip(-d / np.float32(max_depth), np.float32(0.0), np.float32(1.0))
        # Spherical dome profile: steeper near silhouette edges, smooth on top
        edge_slope = np.sqrt(np.clip(np.float32(1.0) - depth * depth * np.float32(0.92), 0.05, 1.0))
        nx = nx2 * edge_slope
        ny = ny2 * edge_slope
        nz = np.sqrt(np.clip(np.float32(1.0) - nx * nx - ny * ny, np.float32(0.04), np.float32(1.0)))
        return nx, ny, nz, depth

    @staticmethod
    def voronoi_field(
        X: np.ndarray,
        Y: np.ndarray,
        cell_size: float = 24.0,
        seed: int = 19,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast vectorized 2D Voronoi cellular noise returning (F1 distance, cell_id_hash)
        for animating organic chromatophore camouflage reveals.
        """
        inv = np.float32(1.0 / cell_size)
        gx = X * inv
        gy = Y * inv
        ix = np.floor(gx).astype(np.int32)
        iy = np.floor(gy).astype(np.int32)
        fx = gx - ix
        fy = gy - iy

        f1 = np.full_like(gx, 8.0, dtype=np.float32)
        cid = np.zeros_like(gx, dtype=np.float32)

        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                cx = (ix + ox).astype(np.float32)
                cy = (iy + oy).astype(np.float32)
                # Overflow-safe deterministic GPU-style fractional spatial hash
                s1 = np.sin(cx * np.float32(127.1) + cy * np.float32(311.7) + np.float32(seed * 17.3)) * np.float32(43758.5453)
                s2 = np.sin(cx * np.float32(269.5) + cy * np.float32(183.3) + np.float32(seed * 29.7)) * np.float32(43758.5453)
                rx = s1 - np.floor(s1)
                ry = s2 - np.floor(s2)
                dist = np.hypot(np.float32(ox) + rx - fx, np.float32(oy) + ry - fy)
                better = dist < f1
                f1 = np.where(better, dist, f1)
                cid = np.where(better, rx * np.float32(0.6) + ry * np.float32(0.4), cid)
        return f1, cid

    @staticmethod
    def directional_feather_noise(
        X: np.ndarray,
        Y: np.ndarray,
        cx: float,
        cy: float,
        angle_rad: float,
    ) -> np.ndarray:
        """
        Directional high-frequency anisotropic noise aligned along wing feather
        barbs to synthesize realistic striated golden eagle plumage.
        """
        dx = X - np.float32(cx)
        dy = Y - np.float32(cy)
        c, s = np.float32(math.cos(angle_rad)), np.float32(math.sin(angle_rad))
        u = dx * c - dy * s
        v = dx * s + dy * c
        # High frequency across barbs (v), slow flow along feather shafts (u)
        w1 = np.sin(v * np.float32(0.72) + np.sin(u * np.float32(0.05)) * np.float32(3.5))
        w2 = np.sin(v * np.float32(1.65) - u * np.float32(0.11)) * np.float32(0.55)
        w3 = np.sin(v * np.float32(3.10) + u * np.float32(0.24)) * np.float32(0.25)
        # Subtle cross-barring (raptor plumage bands)
        bars = np.sin(u * np.float32(0.16)) * np.float32(0.30)
        return np.clip((w1 + w2 + w3) * np.float32(0.28) + bars + np.float32(0.52), 0.0, 1.0)

    @staticmethod
    def shade_sss(
        d: np.ndarray,
        nx: np.ndarray,
        ny: np.ndarray,
        nz: np.ndarray,
        light_dir: np.ndarray,
        sss_color_bgr: Tuple[float, float, float] = (0.14, 0.36, 1.00),
        sss_strength: float = 0.85,
        sss_depth: float = 16.0,
    ) -> np.ndarray:
        """
        Fake Subsurface Scattering (SSS) shader: thin backlit edges glow warm
        red/orange where light wraps and transmits through organic flesh.
        """
        # Thinness peaks near silhouette edges (d slightly below 0)
        thin = np.exp(-np.clip(-d, np.float32(0.0), np.float32(60.0)) / np.float32(sss_depth))
        # Backlight & wrap transmission
        ndotl = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
        wrap = np.clip((ndotl + np.float32(0.55)) / np.float32(1.55), 0.0, 1.0)
        back_scatter = np.clip(-ndotl * np.float32(0.65) + np.float32(0.45), 0.0, 1.0)
        fresnel = np.power(np.clip(np.float32(1.0) - nz, 0.0, 1.0), np.float32(1.9))
        glow = thin * (np.float32(0.45) * wrap + np.float32(0.75) * back_scatter + np.float32(0.65) * fresnel)
        glow = np.clip(glow * np.float32(sss_strength), 0.0, 1.35)
        return glow[..., None] * np.array(sss_color_bgr, dtype=np.float32)[None, None, :]

    @staticmethod
    def shade_iridescence(
        nx: np.ndarray,
        ny: np.ndarray,
        nz: np.ndarray,
        light_dir: np.ndarray,
        phase_shift: float = 0.0,
    ) -> np.ndarray:
        """
        View-angle Iridescence shader using dot(N, V) and half-vector specular
        interference to shimmer richly between metallic emerald green and violet-purple.
        Returns BGR float32 array in [0, 1].
        """
        # Camera vector V = (0, 0, 1) -> dot(N, V) = nz
        ndotv = np.clip(nz, np.float32(0.0), np.float32(1.0))
        # Half-vector H = normalize(L + V)
        hx = light_dir[0]
        hy = light_dir[1]
        hz = light_dir[2] + np.float32(1.0)
        hlen = np.float32(math.sqrt(float(hx * hx + hy * hy + hz * hz)) + 1e-6)
        ndoth = np.clip((nx * hx + ny * hy + nz * hz) / hlen, 0.0, 1.0)

        # Thin-film optical interference phase shifting emerald <-> cyan <-> violet-purple
        phi = (
            (np.float32(1.0) - ndotv) * np.float32(3.6)
            + ndoth * np.float32(1.4)
            + np.float32(phase_shift)
        )
        # BGR palette cycling vividly: emerald (0.35, 0.92, 0.12) <-> violet (0.92, 0.18, 0.68)
        t_mix = np.float32(0.5) + np.float32(0.5) * np.sin(phi)
        emerald_bgr = np.array([0.36, 0.88, 0.08], dtype=np.float32)
        teal_bgr = np.array([0.78, 0.72, 0.08], dtype=np.float32)
        violet_bgr = np.array([0.88, 0.14, 0.62], dtype=np.float32)

        # Smooth 3-stop blend: emerald -> peacock teal -> royal violet
        mid = np.clip(t_mix * np.float32(2.0), 0.0, 1.0)[..., None]
        hi = np.clip((t_mix - np.float32(0.5)) * np.float32(2.0), 0.0, 1.0)[..., None]
        col = emerald_bgr * (1.0 - mid) + teal_bgr * mid
        col = col * (1.0 - hi) + violet_bgr * hi

        # Diffuse + glossy chitin specular + Fresnel rim boost
        ndotl = np.clip(nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2], 0.0, 1.0)
        spec = np.power(ndoth, np.float32(36.0)) * np.float32(1.25)
        rim = np.power(np.float32(1.0) - ndotv, np.float32(2.6)) * np.float32(0.45)

        lit = col * (np.float32(0.32) + np.float32(0.85) * ndotl)[..., None]
        lit += spec[..., None] * np.array([0.95, 1.0, 0.90], dtype=np.float32)
        lit += rim[..., None] * np.array([0.92, 0.35, 0.82], dtype=np.float32)
        return np.clip(lit, 0.0, 1.3)

    @staticmethod
    def composite_sdf(
        canvas: np.ndarray,
        heat_map: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        d: np.ndarray,
        color_bgr: np.ndarray,
        heat_value: Optional[np.ndarray | float] = None,
        shadow_offset: Tuple[int, int] = (0, 0),
        shadow_alpha: float = 0.0,
    ) -> None:
        """
        Anti-aliased alpha composite of an SDF shaded patch (and optional thermal
        heat signature) onto the main float32 canvas.
        """
        x0, y0, x1, y1 = bbox
        H_box, W_box = y1 - y0, x1 - x0
        if H_box <= 0 or W_box <= 0:
            return

        alpha = smoothstep_arr(1.6, -1.6, d)
        if color_bgr.shape[0] != H_box or color_bgr.shape[1] != W_box:
            color_bgr = cv2.resize(color_bgr, (W_box, H_box), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (W_box, H_box), interpolation=cv2.INTER_LINEAR)
            if isinstance(heat_value, np.ndarray):
                heat_value = cv2.resize(heat_value, (W_box, H_box), interpolation=cv2.INTER_LINEAR)

        # Optional soft cast shadow offset behind the entity
        if shadow_alpha > 0.01 and (shadow_offset[0] != 0 or shadow_offset[1] != 0):
            sx, sy = shadow_offset
            sx0 = max(0, x0 + sx)
            sy0 = max(0, y0 + sy)
            sx1 = min(canvas.shape[1], x1 + sx)
            sy1 = min(canvas.shape[0], y1 + sy)
            if sx1 > sx0 and sy1 > sy0:
                ax0, ay0 = sx0 - (x0 + sx), sy0 - (y0 + sy)
                ax1, ay1 = ax0 + (sx1 - sx0), ay0 + (sy1 - sy0)
                sh_a = cv2.GaussianBlur(alpha[ay0:ay1, ax0:ax1], (0, 0), 4.0) * np.float32(shadow_alpha)
                canvas[sy0:sy1, sx0:sx1] *= (np.float32(1.0) - sh_a[..., None])

        dst = canvas[y0:y1, x0:x1]
        a3 = alpha[..., None]
        canvas[y0:y1, x0:x1] = dst * (np.float32(1.0) - a3) + color_bgr * a3

        if heat_map is not None and heat_value is not None:
            hdst = heat_map[y0:y1, x0:x1]
            if isinstance(heat_value, np.ndarray):
                heat_map[y0:y1, x0:x1] = np.maximum(hdst, heat_value * alpha)
            else:
                heat_map[y0:y1, x0:x1] = np.maximum(hdst, np.float32(heat_value) * alpha)


# =============================================================================
# MODULE 2: Lighting_Engine (time_of_day GI, Volumetrics & Thermal Post-FX)
# =============================================================================


@dataclass
class LightingState:
    time_of_day: float               # 0.0 (midnight) -> 0.5 (midday) -> 1.0 (midnight)
    sky_top_bgr: np.ndarray
    sky_horizon_bgr: np.ndarray
    ambient_bgr: np.ndarray
    sun_color_bgr: np.ndarray
    light_dir: np.ndarray            # Normalized 3D key light direction (x, y, z)
    sun_screen_pos: Tuple[float, float]
    sun_intensity: float
    god_ray_strength: float
    rain_intensity: float
    bio_glow_strength: float
    thermal_mix: float               # 0.0 optical vision -> 1.0 thermal FLIR vision
    shadow_darkness: float
    fade_black: float                # 1.0 normal -> 0.0 pitch black fadeout


class Lighting_Engine:
    """
    Dynamic Global Illumination, Weather, Volumetric God Rays, Parallax Jungle
    Environment, and FLIR Thermal Heatmap Post-Processing Engine.
    """

    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        self.width = width
        self.height = height
        self.thermal_lut = self._build_thermal_lut()
        self.cloud_noise = self._build_cloud_texture(320, 180)
        self.bark_tex = self._build_bark_texture(512, 256)

    @staticmethod
    def _build_thermal_lut() -> np.ndarray:
        """Build 256-entry Ironbow / FLIR Thermal Heatmap BGR float32 LUT."""
        # Stops: (pos, R, G, B) -> stored as BGR float32 in [0, 1]
        stops = [
            (0.00, 6, 10, 48),       # Deep cold midnight blue
            (0.18, 18, 28, 128),     # Cold indigo blue
            (0.36, 98, 20, 152),     # Cool violet-purple
            (0.54, 196, 28, 92),     # Warm crimson-magenta
            (0.72, 244, 92, 18),     # Hot vivid red-orange
            (0.88, 255, 212, 34),    # Blazing yellow
            (1.00, 255, 255, 232),   # Incandescent thermal white-yellow
        ]
        lut = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            u = i / 255.0
            for k in range(len(stops) - 1):
                p0, r0, g0, b0 = stops[k]
                p1, r1, g1, b1 = stops[k + 1]
                if p0 <= u <= p1:
                    f = (u - p0) / max(1e-6, p1 - p0)
                    lut[i, 0] = lerp(b0, b1, f) / 255.0
                    lut[i, 1] = lerp(g0, g1, f) / 255.0
                    lut[i, 2] = lerp(r0, r1, f) / 255.0
                    break
        return lut

    @staticmethod
    def _build_cloud_texture(w: int, h: int) -> np.ndarray:
        """Multi-octave procedural FBM atmospheric cloud/mist texture."""
        rng = np.random.default_rng(909)
        acc = np.zeros((h, w), dtype=np.float32)
        amp = 0.55
        for freq in (8, 16, 32, 64):
            raw = rng.uniform(0.0, 1.0, (max(2, h // freq), max(2, w // freq))).astype(np.float32)
            acc += cv2.resize(raw, (w, h), interpolation=cv2.INTER_CUBIC) * amp
            amp *= 0.52
        acc -= acc.min()
        acc /= max(1e-5, float(acc.max()))
        return acc

    @staticmethod
    def _build_bark_texture(w: int, h: int) -> np.ndarray:
        """Procedural organic gnarled tree-bark texture for branch & chameleon camo."""
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ridge = (
            np.sin(yy * 0.24 + np.sin(xx * 0.035) * 4.2) * 0.45
            + np.sin(yy * 0.58 - xx * 0.05) * 0.25
            + np.cos(xx * 0.09 + yy * 0.14) * 0.18
        )
        rng = np.random.default_rng(404)
        grain = cv2.GaussianBlur(rng.normal(0.0, 0.15, (h, w)).astype(np.float32), (0, 0), 1.2)
        val = np.clip(0.50 + ridge * 0.32 + grain, 0.0, 1.0)
        # Mossy bark BGR palette: warm umber brown with hints of muted lichen green
        moss = np.clip(np.sin(xx * 0.045) * np.cos(yy * 0.07) * 1.4, 0.0, 1.0)
        b = 0.11 + 0.14 * val
        g = 0.15 + 0.18 * val + 0.11 * moss
        r = 0.19 + 0.22 * val - 0.04 * moss
        return np.stack((b, g, r), axis=-1).astype(np.float32)

    def evaluate(self, t: float) -> LightingState:
        """Compute global illumination, weather, sun angle & thermal state at time t."""
        # Keyframe table across 120s matching the 7-act emotional & diurnal arc:
        # (time, tod, sky_top_bgr, sky_hor_bgr, sun_bgr, amb_bgr, sun_int, rays, rain, bio, shadow)
        keys = [
            (0.0,   0.02, (0.09, 0.05, 0.03), (0.16, 0.10, 0.06), (0.45, 0.42, 0.30), (0.14, 0.10, 0.06), 0.10, 0.00, 1.00, 1.00, 0.10),
            (14.0,  0.12, (0.12, 0.08, 0.05), (0.22, 0.14, 0.09), (0.50, 0.48, 0.38), (0.16, 0.12, 0.08), 0.14, 0.05, 0.92, 0.95, 0.10),
            (20.0,  0.24, (0.34, 0.26, 0.32), (0.35, 0.58, 0.92), (0.42, 0.78, 1.00), (0.26, 0.34, 0.44), 0.78, 0.72, 0.00, 0.22, 0.00),
            (30.0,  0.36, (0.58, 0.48, 0.26), (0.64, 0.78, 0.88), (0.72, 0.94, 1.00), (0.34, 0.44, 0.40), 0.96, 0.55, 0.00, 0.00, 0.00),
            (48.0,  0.52, (0.66, 0.54, 0.24), (0.68, 0.82, 0.76), (0.82, 0.96, 1.00), (0.36, 0.48, 0.42), 1.00, 0.45, 0.00, 0.00, 0.00),
            (66.0,  0.65, (0.44, 0.40, 0.24), (0.45, 0.64, 0.76), (0.58, 0.84, 1.00), (0.28, 0.36, 0.34), 0.88, 0.40, 0.00, 0.00, 0.05),
            (72.5,  0.72, (0.12, 0.10, 0.06), (0.16, 0.16, 0.12), (0.28, 0.36, 0.40), (0.10, 0.12, 0.10), 0.22, 0.05, 0.00, 0.00, 0.65),
            (89.5,  0.76, (0.12, 0.10, 0.06), (0.18, 0.16, 0.12), (0.30, 0.38, 0.42), (0.10, 0.12, 0.10), 0.25, 0.05, 0.00, 0.00, 0.60),
            (92.5,  0.82, (0.24, 0.22, 0.48), (0.16, 0.44, 0.98), (0.22, 0.56, 1.00), (0.18, 0.26, 0.46), 1.05, 0.92, 0.00, 0.00, 0.00),
            (108.0, 0.88, (0.18, 0.14, 0.34), (0.12, 0.30, 0.82), (0.18, 0.42, 0.96), (0.14, 0.18, 0.34), 0.82, 0.78, 0.00, 0.00, 0.08),
            (114.5, 0.97, (0.04, 0.02, 0.02), (0.07, 0.05, 0.04), (0.20, 0.20, 0.22), (0.06, 0.05, 0.04), 0.06, 0.00, 0.45, 0.95, 0.20),
            (120.0, 1.00, (0.02, 0.01, 0.01), (0.04, 0.03, 0.02), (0.18, 0.18, 0.20), (0.05, 0.04, 0.03), 0.04, 0.00, 0.55, 1.00, 0.25),
        ]

        # Piecewise smooth interpolation
        idx = 0
        for i in range(len(keys) - 1):
            if keys[i][0] <= t <= keys[i + 1][0]:
                idx = i
                break
        k0, k1 = keys[idx], keys[idx + 1]
        u = smoothstep((t - k0[0]) / max(1e-5, k1[0] - k0[0]))

        tod = lerp(k0[1], k1[1], u)
        sky_top = lerp_vec(k0[2], k1[2], u)
        sky_hor = lerp_vec(k0[3], k1[3], u)
        sun_col = lerp_vec(k0[4], k1[4], u)
        amb_col = lerp_vec(k0[5], k1[5], u)
        sun_int = lerp(k0[6], k1[6], u)
        rays = lerp(k0[7], k1[7], u)
        rain = lerp(k0[8], k1[8], u)
        bio = lerp(k0[9], k1[9], u)
        shadow = lerp(k0[10], k1[10], u)

        # Sun arc across sky driven by time_of_day
        sun_arc = (tod - 0.18) / 0.72  # 0 at dawn -> 0.5 midday -> 1.0 sunset
        sx = self.width * (0.15 + 0.72 * clamp(sun_arc, -0.2, 1.2))
        sy = self.height * (0.58 - 0.46 * math.sin(math.pi * clamp(sun_arc, 0.0, 1.0)))
        # 3D directional light pointing TOWARD light source for N·L shading
        lx = (sx - self.width * 0.5) / (self.width * 0.6)
        ly = (sy - self.height * 0.55) / (self.height * 0.6)
        lz = 0.55 if (0.2 < tod < 0.85) else 0.35
        lvec = np.array([lx, ly, lz], dtype=np.float32)
        lvec /= float(np.linalg.norm(lvec) + 1e-6)

        # Act V Thermal Heatmap transition (73.5s -> 74.5s ON, 89.2s -> 90.2s OFF)
        if 73.5 <= t <= 90.2:
            th_in = smoothstep((t - 73.5) / 1.0)
            th_out = 1.0 - smoothstep((t - 89.2) / 1.0)
            thermal_mix = th_in * th_out
        else:
            thermal_mix = 0.0

        # Smooth fade to black at end of Act VII (118.2s -> 120.0s)
        fade_black = 1.0 - smoothstep((t - 118.2) / 1.75) if t > 118.2 else 1.0
        # Subtle fade-in from pitch black at very opening (0.0s -> 0.8s)
        if t < 0.8:
            fade_black *= smoothstep(t / 0.8)

        return LightingState(
            time_of_day=tod,
            sky_top_bgr=sky_top,
            sky_horizon_bgr=sky_hor,
            ambient_bgr=amb_col,
            sun_color_bgr=sun_col,
            light_dir=lvec,
            sun_screen_pos=(sx, sy),
            sun_intensity=sun_int,
            god_ray_strength=rays,
            rain_intensity=rain,
            bio_glow_strength=bio,
            thermal_mix=thermal_mix,
            shadow_darkness=shadow,
            fade_black=fade_black,
        )

    def render_environment(
        self,
        t: float,
        light: LightingState,
        cam_x: float,
        cam_y: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render volumetric sky, sun/moon glow, crepuscular god rays, parallax
        canopy silhouettes, and wet jungle soil floor. Returns (bgr_float32, heat_map).
        """
        sw, sh = 320, 180
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        ny = yy / float(sh - 1)

        # 1. Atmospheric vertical sky gradient + drifting FBM storm/sunset clouds
        grad_u = np.power(np.clip(ny * 1.18, 0.0, 1.0), 1.35)[..., None]
        sky = light.sky_top_bgr * (1.0 - grad_u) + light.sky_horizon_bgr * grad_u

        shift_x = int(round(t * 2.5 + cam_x * 0.08)) % sw
        clouds = np.roll(self.cloud_noise, shift=-shift_x, axis=1)
        cloud_tint = light.sun_color_bgr * 0.45 + light.sky_horizon_bgr * 0.55
        cloud_mask = np.clip((clouds - 0.35) * 1.6, 0.0, 1.0) * (1.0 - ny * 0.75)
        sky = sky * (1.0 - cloud_mask[..., None] * 0.38) + cloud_tint * (cloud_mask[..., None] * 0.38)

        # 2. Sun / Golden-Hour atmospheric radial halo
        sx = (light.sun_screen_pos[0] - cam_x * 0.15) * (sw / float(self.width))
        sy = (light.sun_screen_pos[1] - cam_y * 0.15) * (sh / float(self.height))
        sdist = np.hypot((xx - sx) * 0.9, yy - sy)
        halo = np.exp(-sdist / 42.0) * 0.65 + np.exp(-sdist / 12.0) * 0.85
        sky += light.sun_color_bgr * (halo * light.sun_intensity * 0.72)[..., None]

        # 3. Volumetric crepuscular god rays piercing canopy gaps
        if light.god_ray_strength > 0.02:
            ang = np.arctan2(yy - sy + 8.0, xx - sx + 1e-4)
            shafts = (
                0.55 * np.sin(ang * 14.0 + 0.4 + t * 0.15)
                + 0.45 * np.sin(ang * 25.0 - 1.1)
            )
            shafts = np.power(np.clip(shafts, 0.0, 1.0), 2.2)
            falloff = np.exp(-sdist / 155.0) * (0.25 + 0.75 * ny)
            sky += light.sun_color_bgr * (shafts * falloff * light.god_ray_strength * 0.34)[..., None]

        # Upscale smooth atmospheric backdrop to full 1280x720
        canvas = cv2.resize(np.clip(sky, 0.0, 1.4), (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # 4. Lush soft-focus parallax rainforest background (varied leaning trunks, arching canopy & vines)
        bg_sil = np.zeros((sh, sw), dtype=np.float32)
        trunks = [
            (22, 16, -6), (78, 10, 5), (146, 22, -4), (218, 12, 8),
            (278, 18, -7), (355, 24, 5), (430, 14, -5),
        ]
        for tx_w, tw, lean in trunks:
            px = int(round(tx_w - cam_x * 0.09))
            if -45 <= px <= sw + 45:
                pts = np.array(
                    [
                        [px - tw, sh],
                        [px - int(tw * 0.65) + lean, 0],
                        [px + int(tw * 0.65) + lean, 0],
                        [px + tw, sh],
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(bg_sil, [pts], 0.72, lineType=cv2.LINE_AA)
                # Natural diagonal boughs branching upward
                cv2.line(bg_sil, (px + lean // 2, 48), (px + lean * 3 + 28, 10), 0.65, max(2, tw // 3), cv2.LINE_AA)
                cv2.line(bg_sil, (px + lean // 2, 62), (px + lean * 2 - 26, 20), 0.60, max(2, tw // 4), cv2.LINE_AA)

        # Layered organic arching rainforest canopy foliage across upper frame
        for idx in range(14):
            cx_c = int(round((idx * 34 - cam_x * 0.11) % (sw + 70) - 35))
            cy_c = int(10 + 18 * math.sin(idx * 1.7))
            rx_c = 28 + (idx * 13) % 24
            ry_c = 18 + (idx * 7) % 14
            cv2.ellipse(bg_sil, (cx_c, cy_c), (rx_c, ry_c), (idx * 17) % 35 - 17, 0, 360, 0.85, -1, cv2.LINE_AA)

        # Delicate hanging jungle lianas / vines (catenary curves)
        for v_i, (vx0, vx1, sag) in enumerate([(30, 115, 58), (135, 235, 72), (245, 345, 62)]):
            vpts = []
            for k in range(12):
                u_v = k / 11.0
                vx = lerp(vx0, vx1, u_v) - cam_x * 0.10
                vy = 8.0 + sag * 4.0 * u_v * (1.0 - u_v)
                vpts.append([int(vx), int(vy)])
            cv2.polylines(bg_sil, [np.array(vpts, dtype=np.int32)], False, 0.55, 1, cv2.LINE_AA)

        # Soft atmospheric depth-of-field blur on distant rainforest canopy!
        bg_sil = cv2.GaussianBlur(bg_sil, (0, 0), 2.1)
        sil_full = cv2.resize(bg_sil, (self.width, self.height), interpolation=cv2.INTER_LINEAR)[..., None]
        sil_col = (light.ambient_bgr * 0.36 + light.sky_top_bgr * 0.28)[None, None, :]
        canvas = canvas * (np.float32(1.0) - sil_full * np.float32(0.78)) + sil_col * (sil_full * np.float32(0.78))

        # 5. Rich dark wet jungle soil floor with organic rolling terrain contour
        base_gy = 558.0 - cam_y * 0.85
        g_xx = np.arange(self.width, dtype=np.float32) + np.float32(cam_x)
        terrain_y = (
            np.float32(base_gy)
            + np.float32(14.0) * np.sin(g_xx * np.float32(0.0055))
            + np.float32(7.0) * np.cos(g_xx * np.float32(0.016) + np.float32(0.9))
        )
        gy_min = max(0, int(np.floor(float(terrain_y.min()))) - 4)
        if gy_min < self.height:
            gh = self.height - gy_min
            yy_g = np.arange(gy_min, self.height, dtype=np.float32)[:, None]
            depth_below = yy_g - terrain_y[None, :]
            g_ny = np.clip(depth_below / np.float32(max(1, self.height - gy_min)), 0.0, 1.0)
            soil_tex = (
                np.float32(0.5)
                + np.float32(0.24) * np.sin(g_xx[None, :] * np.float32(0.022) + g_ny * np.float32(10.0))
                + np.float32(0.14) * np.cos(g_xx[None, :] * np.float32(0.068) - g_ny * np.float32(16.0))
            )
            soil_base = (
                np.array([0.06, 0.08, 0.10], dtype=np.float32) * (np.float32(0.45) + np.float32(0.9) * light.ambient_bgr)
                + np.array([0.02, 0.035, 0.045], dtype=np.float32) * soil_tex[..., None]
            )
            if light.rain_intensity > 0.05 or light.bio_glow_strength > 0.05:
                wet = np.exp(-((g_ny - np.float32(0.32)) / np.float32(0.34)) ** 2) * np.float32(
                    0.30 * light.rain_intensity + 0.16 * light.bio_glow_strength
                )
                soil_base += light.sky_horizon_bgr * wet[..., None] * np.float32(0.65)

            edge_blend = smoothstep_arr(-2.0, 12.0, depth_below)[..., None]
            canvas[gy_min:, :] = canvas[gy_min:, :] * (np.float32(1.0) - edge_blend) + soil_base * edge_blend

        # 6. Deep canopy shadow darkening (Act V transition)
        if light.shadow_darkness > 0.01:
            canvas *= np.float32(1.0 - 0.62 * light.shadow_darkness)

        # Initialize cold thermal background heat map (only populated actively near Act V)
        heat_map = None
        if light.thermal_mix > 0.005:
            # Cold ambient canopy/sky heat field (~0.04 to 0.17 -> deep cold blue in Ironbow LUT)
            cold_var = cv2.resize(clouds, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            heat_map = (np.float32(0.05) + cold_var * np.float32(0.11)).astype(np.float32)

        return canvas, heat_map

    def apply_thermal_filter(
        self,
        canvas: np.ndarray,
        heat_map: np.ndarray,
        thermal_mix: float,
        t: float,
    ) -> np.ndarray:
        """
        Custom FLIR Thermal Heatmap Vision post-processing filter applied to the
        entire NumPy array: background goes dark cold blue while warm reptiles
        blaze bright red/yellow/white.
        """
        if thermal_mix <= 0.005:
            return canvas

        # Add subtle thermal bloom around hot biological heat signatures
        hot_only = np.clip((heat_map - np.float32(0.45)) * np.float32(1.8), 0.0, 1.0)
        hot_small = cv2.resize(hot_only, (320, 180), interpolation=cv2.INTER_AREA)
        hot_blur = cv2.GaussianBlur(hot_small, (0, 0), 5.5)
        bloom = cv2.resize(hot_blur, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Subtle thermal sensor horizontal scanline micro-shimmer
        scan = (np.sin(np.arange(self.height, dtype=np.float32) * 1.5 + t * 8.0) * 0.012)[:, None]
        heat_final = np.clip(heat_map + bloom * np.float32(0.28) + scan, 0.0, 0.999)
        idx = (heat_final * np.float32(255.0)).astype(np.uint8)
        thermal_bgr = self.thermal_lut[idx]

        # Subtle predator optical thermal vignette
        yy, xx = np.mgrid[0:self.height:4, 0:self.width:4].astype(np.float32)
        nx = (xx - self.width * 0.5) / (self.width * 0.52)
        ny = (yy - self.height * 0.5) / (self.height * 0.52)
        vig_s = np.clip(1.0 - (nx * nx + ny * ny - 0.55) * 0.42, 0.55, 1.0)
        vig = cv2.resize(vig_s, (self.width, self.height), interpolation=cv2.INTER_LINEAR)[..., None]
        thermal_bgr *= vig

        m = np.float32(thermal_mix)
        return canvas * (np.float32(1.0) - m) + thermal_bgr * m


# =============================================================================
# MODULE 3: Particle_System (Deterministic Vectorized Physics Engine)
# =============================================================================


class Particle_System:
    """
    Lightweight, deterministic, vectorized NumPy particle physics engine
    rendering rain streaks, expanding concentric water ripples, cyan fungal
    spores, sunlit floating pollen, and 3D perspective-projected falling
    feathers exploding toward the lens.
    """

    def __init__(self, width: int = WIDTH, height: int = HEIGHT, seed: int = 2026):
        self.width = width
        self.height = height
        rng = np.random.default_rng(seed)

        # 1. Raindrops (260 vectorized streaks)
        n_rain = 260
        self.rain_x0 = rng.uniform(-120.0, width + 180.0, n_rain).astype(np.float32)
        self.rain_y0 = rng.uniform(0.0, height, n_rain).astype(np.float32)
        self.rain_speed = rng.uniform(680.0, 1080.0, n_rain).astype(np.float32)
        self.rain_len = rng.uniform(20.0, 44.0, n_rain).astype(np.float32)
        self.rain_alpha = rng.uniform(0.18, 0.52, n_rain).astype(np.float32)

        # 2. Deterministic water ripple splash sites (shared with DSP_Audio_Synth!)
        n_rip = 42
        self.rip_x = rng.uniform(70.0, width - 70.0, n_rip).astype(np.float32)
        self.rip_y = rng.uniform(575.0, height - 22.0, n_rip).astype(np.float32)
        self.rip_period = rng.uniform(1.65, 3.15, n_rip).astype(np.float32)
        self.rip_phase = rng.uniform(0.0, 3.0, n_rip).astype(np.float32)
        self.rip_max_r = rng.uniform(28.0, 62.0, n_rip).astype(np.float32)

        # 3. Bioluminescent spores (Acts I & VII)
        n_sp = 75
        self.sp_x = rng.uniform(80.0, width - 80.0, n_sp).astype(np.float32)
        self.sp_y0 = rng.uniform(430.0, 660.0, n_sp).astype(np.float32)
        self.sp_vy = rng.uniform(14.0, 34.0, n_sp).astype(np.float32)
        self.sp_amp = rng.uniform(14.0, 38.0, n_sp).astype(np.float32)
        self.sp_freq = rng.uniform(0.7, 1.9, n_sp).astype(np.float32)
        self.sp_phase = rng.uniform(0.0, 6.28, n_sp).astype(np.float32)

        # 4. Sunlit Golden Pollen (Acts II, III, IV)
        n_pol = 135
        self.pol_x0 = rng.uniform(-100.0, width + 550.0, n_pol).astype(np.float32)
        self.pol_y0 = rng.uniform(40.0, 640.0, n_pol).astype(np.float32)
        self.pol_vx = rng.uniform(8.0, 26.0, n_pol).astype(np.float32)
        self.pol_amp = rng.uniform(18.0, 48.0, n_pol).astype(np.float32)
        self.pol_freq = rng.uniform(0.35, 1.15, n_pol).astype(np.float32)
        self.pol_phase = rng.uniform(0.0, 6.28, n_pol).astype(np.float32)
        self.pol_r = rng.uniform(1.6, 4.4, n_pol).astype(np.float32)

        # 5. Act VI 3D Eagle Impact Feather Explosion (95 feathers drifting toward camera lens!)
        n_f = 95
        ang = rng.uniform(0.0, 2.0 * math.pi, n_f)
        spd = rng.uniform(90.0, 520.0, n_f)
        self.f_vx = (np.cos(ang) * spd).astype(np.float32)
        self.f_vy = (np.sin(ang) * spd - rng.uniform(80.0, 290.0, n_f)).astype(np.float32)
        # Positive vz moves feathers TOWARD camera lens (z_cam = 520)
        self.f_vz = rng.uniform(45.0, 240.0, n_f).astype(np.float32)
        self.f_drag = rng.uniform(0.85, 1.65, n_f).astype(np.float32)
        self.f_spin = rng.uniform(-3.6, 3.6, n_f).astype(np.float32)
        self.f_rot0 = rng.uniform(0.0, 6.28, n_f).astype(np.float32)
        self.f_len = rng.uniform(22.0, 48.0, n_f).astype(np.float32)
        self.f_sway_ph = rng.uniform(0.0, 6.28, n_f).astype(np.float32)
        self.f_shade = rng.uniform(0.55, 1.0, n_f).astype(np.float32)

    def get_water_drop_events(self, t_start: float, t_end: float) -> List[Tuple[float, float]]:
        """Return deterministic list of (impact_time, screen_x) for binaural audio pings."""
        events: List[Tuple[float, float]] = []
        # Use a subset of 14 prominent drop streams so audio sounds crisp & musical
        for i in range(0, len(self.rip_x), 3):
            per = float(self.rip_period[i])
            ph = float(self.rip_phase[i])
            x = float(self.rip_x[i])
            k_min = int(math.ceil((t_start + ph) / per))
            k_max = int(math.floor((t_end + ph) / per))
            for k in range(k_min, k_max + 1):
                et = k * per - ph
                if t_start <= et <= t_end:
                    events.append((et, x))
        return sorted(events)

    def render_rain_and_ripples(
        self, canvas: np.ndarray, t: float, intensity: float, bio_glow: float
    ) -> None:
        """Render vectorized slanted rain streaks + concentric expanding water ripples."""
        if intensity <= 0.01:
            return

        # 1. Expanding concentric water ripples (`cv2.circle` / ellipse with fading alpha)
        ages = (t + self.rip_phase) % self.rip_period
        life = np.float32(1.15)
        active = ages < life
        for idx in np.flatnonzero(active):
            u = float(ages[idx] / life)
            alpha = ((1.0 - u) ** 1.6) * intensity
            cx, cy = int(self.rip_x[idx]), int(self.rip_y[idx])
            r_outer = max(2, int(self.rip_max_r[idx] * (0.12 + 0.88 * math.sqrt(u))))
            col = (
                (0.95 * bio_glow + 0.55 * (1.0 - bio_glow)) * alpha,
                (0.85 * bio_glow + 0.52 * (1.0 - bio_glow)) * alpha,
                (0.28 * bio_glow + 0.45 * (1.0 - bio_glow)) * alpha,
            )
            # Draw directly on minimal ROI around ripple for zero full-frame allocations
            pad = r_outer + 8
            x0, y0 = max(0, cx - pad), max(0, cy - pad)
            x1, y1 = min(self.width, cx + pad), min(self.height, cy + pad)
            if x1 > x0 and y1 > y0:
                roi = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
                lcx, lcy = cx - x0, cy - y0
                cv2.ellipse(roi, (lcx, lcy), (r_outer, max(1, int(r_outer * 0.34))), 0, 0, 360, col, 2, cv2.LINE_AA)
                if r_outer > 10:
                    r_in = int(r_outer * 0.56)
                    cv2.ellipse(
                        roi, (lcx, lcy), (r_in, max(1, int(r_in * 0.34))), 0, 0, 360,
                        (col[0] * 0.65, col[1] * 0.65, col[2] * 0.65), 1, cv2.LINE_AA
                    )
                if u < 0.18:
                    splash_r = max(1, int(5.0 * (1.0 - u / 0.18)))
                    cv2.circle(roi, (lcx, max(0, lcy - int(u * 35))), splash_r, col, -1, cv2.LINE_AA)
                canvas[y0:y1, x0:x1] += roi * np.float32(0.95)

        # 2. Vectorized slanted rain streaks drawn onto half-res additive overlay (super fast!)
        hw, hh = self.width // 2, self.height // 2
        rain_half = np.zeros((hh, hw, 3), dtype=np.float32)
        wind_vx = np.float32(-145.0)
        rx = ((self.rain_x0 + wind_vx * np.float32(t)) % np.float32(self.width + 240.0) - np.float32(120.0)) * np.float32(0.5)
        ry = ((self.rain_y0 + self.rain_speed * np.float32(t)) % np.float32(self.height + 60.0) - np.float32(30.0)) * np.float32(0.5)
        n_draw = int(len(rx) * clamp(intensity))
        for i in range(n_draw):
            x0, y0 = int(rx[i]), int(ry[i])
            L = float(self.rain_len[i]) * 0.5
            x1, y1 = int(x0 - 0.17 * L), int(y0 + L)
            a = float(self.rain_alpha[i]) * intensity
            cv2.line(rain_half, (x0, y0), (x1, y1), (0.90 * a, 0.86 * a, 0.74 * a), 1, cv2.LINE_AA)
        canvas += cv2.resize(rain_half, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def render_bio_spores(
        self, canvas: np.ndarray, t: float, strength: float, cam_x: float
    ) -> None:
        """Render glowing cyan bioluminescent spores rising slowly in Acts I & VII."""
        if strength <= 0.02:
            return
        hw, hh = self.width // 2, self.height // 2
        buf = np.zeros((hh, hw, 3), dtype=np.float32)
        ys = (665.0 - ((665.0 - self.sp_y0 + self.sp_vy * t) % 270.0)) * 0.5
        xs = (self.sp_x - cam_x + self.sp_amp * np.sin(self.sp_freq * t + self.sp_phase)) * 0.5
        pulse = 0.55 + 0.45 * np.sin(t * 2.8 + self.sp_phase * 2.0)
        fade_top = np.clip((ys * 2.0 - 395.0) / 80.0, 0.0, 1.0)
        for i in range(len(xs)):
            ix, iy = int(xs[i]), int(ys[i])
            if 0 <= ix < hw and 0 <= iy < hh:
                a = float(pulse[i] * fade_top[i]) * strength
                cv2.circle(buf, (ix, iy), 1, (1.0 * a, 0.94 * a, 0.22 * a), -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(buf, (0, 0), 2.8)
        canvas += cv2.resize(buf + glow * np.float32(1.1), (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def render_pollen(
        self, canvas: np.ndarray, t: float, strength: float, cam_x: float, sun_bgr: np.ndarray
    ) -> None:
        """Render warm floating sunlit bokeh pollen particles in Acts II-IV."""
        if strength <= 0.02:
            return
        hw, hh = self.width // 2, self.height // 2
        buf = np.zeros((hh, hw, 3), dtype=np.float32)
        xs = (((self.pol_x0 + self.pol_vx * t) % (self.width + 650.0)) - 80.0 - cam_x * 0.7) * 0.5
        ys = (self.pol_y0 + self.pol_amp * np.sin(self.pol_freq * t + self.pol_phase)) * 0.5
        twinkle = 0.5 + 0.5 * np.sin(t * 2.1 + self.pol_phase * 1.7)
        for i in range(len(xs)):
            ix, iy = int(xs[i]), int(ys[i])
            if -5 <= ix < hw + 5 and -5 <= iy < hh + 5:
                a = float(twinkle[i]) * strength * 0.78
                col = (float(sun_bgr[0]) * a, float(sun_bgr[1]) * a, float(sun_bgr[2]) * a)
                cv2.circle(buf, (ix, iy), max(1, int(self.pol_r[i] * 0.55)), col, -1, cv2.LINE_AA)
        soft = cv2.GaussianBlur(buf, (0, 0), 2.0)
        canvas += cv2.resize(buf * np.float32(0.75) + soft * np.float32(0.85), (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def render_feather_explosion(
        self,
        canvas: np.ndarray,
        t: float,
        impact_t: float,
        origin: Tuple[float, float],
        sun_bgr: np.ndarray,
    ) -> None:
        """
        Render 95 procedural golden-eagle feathers bursting from the Act VI
        impact and drifting in 3D perspective directly toward the camera lens!
        """
        age = t - impact_t
        if age < 0.0 or age > 12.0:
            return

        fade_out = 1.0 - smoothstep((age - 8.5) / 3.2) if age > 8.5 else 1.0
        if fade_out <= 0.01:
            return

        # Exponential air-drag displacement integral: (1 - exp(-k*age)) / k
        exp_term = (1.0 - np.exp(-self.f_drag * age)) / self.f_drag
        z_cam = np.float32(520.0)
        z = np.float32(60.0) + self.f_vz * exp_term * np.float32(1.45) + np.float32(18.0 * age)
        depth_dist = z_cam - z

        # Gentle gravity + sinusoidal feather flutter
        wx = origin[0] + self.f_vx * exp_term + 38.0 * np.sin(1.8 * age + self.f_sway_ph)
        wy = origin[1] + self.f_vy * exp_term + 28.0 * age * age / (1.0 + 0.4 * age)

        # Sort back-to-front so close-to-lens giant feathers overlap properly
        order = np.argsort(z)
        for idx in order:
            dz = float(depth_dist[idx])
            if dz < 35.0:  # Already flew past camera lens
                continue
            persp = 440.0 / dz
            sx = self.width * 0.5 + (float(wx[idx]) - self.width * 0.5) * persp
            sy = self.height * 0.5 + (float(wy[idx]) - self.height * 0.5) * persp
            flen = float(self.f_len[idx]) * persp
            if sx < -flen * 2 or sx > self.width + flen * 2 or sy < -flen * 2 or sy > self.height + flen * 2:
                continue

            rot = float(self.f_rot0[idx] + self.f_spin[idx] * exp_term[idx] + 0.5 * math.sin(2.4 * age + float(self.f_sway_ph[idx])))
            sh = float(self.f_shade[idx])
            # Fade gently as feather gets extremely close to lens
            near_fade = clamp((dz - 35.0) / 85.0)
            alpha = fade_out * near_fade * 0.92
            self.draw_procedural_feather(
                canvas, (sx, sy), flen, rot, sh, alpha, sun_bgr
            )

    @staticmethod
    def draw_procedural_feather(
        canvas: np.ndarray,
        center: Tuple[float, float],
        length: float,
        angle: float,
        shade: float,
        alpha: float,
        rim_bgr: np.ndarray,
        glow_quill: float = 0.0,
    ) -> None:
        """Draw a detailed procedural golden-eagle feather with curved quill, barbs & SSS rim."""
        if alpha <= 0.01 or length < 4.0:
            return
        cx, cy = center
        ca, sa = math.cos(angle), math.sin(angle)
        half_l = length * 0.5
        half_w = length * 0.19

        # Compute bounding box for fast alpha blending
        rad = int(math.ceil(half_l * 1.35)) + 4
        x0, y0 = max(0, int(cx) - rad), max(0, int(cy) - rad)
        x1, y1 = min(canvas.shape[1], int(cx) + rad), min(canvas.shape[0], int(cy) + rad)
        if x1 <= x0 or y1 <= y0:
            return

        patch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        lcx, lcy = cx - x0, cy - y0

        def pt(u: float, v: float) -> Tuple[int, int]:
            # Subtle natural quill bow curvature
            bow = 0.06 * length * (1.0 - (u / max(1e-4, half_l)) ** 2)
            px = lcx + u * ca - (v + bow) * sa
            py = lcy + u * sa + (v + bow) * ca
            return int(round(px)), int(round(py))

        # Tapered aerodynamic feather vane polygon
        vane_top = [pt(lerp(-half_l * 0.65, half_l, s), -half_w * math.sin(math.pi * (s ** 0.72))) for s in np.linspace(0, 1, 10)]
        vane_bot = [pt(lerp(half_l, -half_l * 0.65, s), half_w * math.sin(math.pi * ((1.0 - s) ** 0.72))) for s in np.linspace(0, 1, 10)]
        poly = np.array(vane_top + vane_bot, dtype=np.int32)

        base_col = (0.12 * shade, 0.23 * shade, 0.38 * shade)  # Golden-brown eagle plumage BGR
        rim_col = (
            float(rim_bgr[0]) * 0.55 + 0.15,
            float(rim_bgr[1]) * 0.55 + 0.25,
            float(rim_bgr[2]) * 0.75 + 0.25,
        )
        cv2.fillPoly(patch, [poly], base_col, lineType=cv2.LINE_AA)
        cv2.fillPoly(mask, [poly], 1.0, lineType=cv2.LINE_AA)

        # Glowing Golden Hour backlit SSS rim along vane perimeter
        cv2.polylines(patch, [poly], True, rim_col, max(1, int(length * 0.035)), lineType=cv2.LINE_AA)

        # Fine procedural directional barb notches
        n_barbs = min(14, max(5, int(length / 5.0)))
        for k in range(1, n_barbs):
            s = k / float(n_barbs)
            u_b = lerp(-half_l * 0.55, half_l * 0.88, s)
            w_b = half_w * math.sin(math.pi * (s ** 0.72))
            bcol = (base_col[0] * 0.58, base_col[1] * 0.58, base_col[2] * 0.58)
            cv2.line(patch, pt(u_b, 0), pt(u_b + length * 0.08, -w_b), bcol, 1, cv2.LINE_AA)
            cv2.line(patch, pt(u_b, 0), pt(u_b + length * 0.08, w_b), bcol, 1, cv2.LINE_AA)

        # Central pale rachis / quill shaft (optionally infused with cyan mycelium glow in Act VII!)
        q_col = (
            lerp(0.65, 1.00, glow_quill),
            lerp(0.76, 0.96, glow_quill),
            lerp(0.84, 0.25, glow_quill),
        )
        q_pts = np.array([pt(lerp(-half_l, half_l * 0.95, s), 0.0) for s in np.linspace(0, 1, 8)], dtype=np.int32)
        cv2.polylines(patch, [q_pts], False, q_col, max(1, int(length * 0.04)), lineType=cv2.LINE_AA)
        cv2.polylines(mask, [q_pts], False, 1.0, max(1, int(length * 0.05)), lineType=cv2.LINE_AA)

        # Soft out-of-focus depth-of-field blur when giant feather rushes right past camera lens
        if length > 140.0:
            sigma = min(5.0, (length - 140.0) * 0.03)
            patch = cv2.GaussianBlur(patch, (0, 0), sigma)
            mask = cv2.GaussianBlur(mask, (0, 0), sigma)

        a3 = (mask * np.float32(alpha))[..., None]
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (np.float32(1.0) - a3) + patch * a3


# =============================================================================
# MODULE 4: DSP_Audio_Synth (Brian Eno Generative Score + Binaural Foley)
# =============================================================================


class DSP_Audio_Synth:
    """
    Pure NumPy + SciPy Audio DSP Engine synthesizing:
      - A Brian Eno-style generative ambient score (5 detuned LFO-modulated
        voices morphing across chords + Schroeder algorithmic reverb array).
      - Binaural stereo-panned foley mapped directly to on-screen X coordinates
        (constant-power stereo pan + interaural time difference ITD).
    """

    def __init__(
        self,
        particles: Particle_System,
        duration: float = DURATION,
        sample_rate: int = SAMPLE_RATE,
        seed: int = 4242,
    ):
        self.particles = particles
        self.duration = duration
        self.sr = sample_rate
        self.total_samples = int(round(duration * sample_rate))
        self.rng = np.random.default_rng(seed)

    def _butter(
        self, sig: np.ndarray, cutoff: float | Tuple[float, float], btype: str, order: int = 4
    ) -> np.ndarray:
        nyq = 0.5 * self.sr
        if isinstance(cutoff, tuple):
            wn = (clamp(cutoff[0] / nyq, 1e-4, 0.995), clamp(cutoff[1] / nyq, 1e-4, 0.995))
        else:
            wn = clamp(cutoff / nyq, 1e-4, 0.995)
        sos = signal.butter(order, wn, btype=btype, output="sos")
        return signal.sosfilt(sos, sig)

    @staticmethod
    def _fade(sig: np.ndarray, sr: int, fi_s: float = 0.008, fo_s: float = 0.015) -> np.ndarray:
        out = sig.copy()
        n = len(out)
        fi = min(n // 2, max(1, int(fi_s * sr)))
        fo = min(n // 2, max(1, int(fo_s * sr)))
        out[:fi] *= np.linspace(0.0, 1.0, fi)
        out[-fo:] *= np.linspace(1.0, 0.0, fo)
        return out

    def _pink_noise(self, n: int) -> np.ndarray:
        """Synthesize true 1/f pink noise via spectral shaping."""
        white = self.rng.standard_normal(n)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sr)
        scale = 1.0 / np.sqrt(np.maximum(freqs, 18.0))
        pink = np.fft.irfft(spec * scale, n=n)
        pink /= float(np.std(pink) + 1e-8)
        return pink

    def _algorithmic_reverb(self, stereo: np.ndarray, wet: float = 0.58) -> np.ndarray:
        """
        Schroeder / Moorer algorithmic stereo reverb using 6 parallel prime-delay
        feedback comb filters + 2 cascaded allpass diffusers for cavernous Eno ambience.
        """
        n = stereo.shape[0]
        out = np.zeros_like(stereo)

        # Decorrelated prime delay comb lengths (samples @ 44.1 kHz) per channel
        ch_combs = [
            [(1687, 0.80), (1901, 0.78), (2207, 0.76), (2503, 0.74), (2819, 0.72), (3137, 0.70)],
            [(1723, 0.80), (1949, 0.78), (2243, 0.76), (2549, 0.74), (2857, 0.72), (3181, 0.70)],
        ]
        allpasses = [(347, 0.62), (521, 0.58)]

        for ch in (0, 1):
            x = stereo[:, ch]
            comb_sum = np.zeros(n, dtype=np.float64)
            for delay, fb in ch_combs[ch]:
                # Exact sparse comb recurrence: y[i] = x[i - D] + fb * y[i - D]
                yc = np.zeros(n, dtype=np.float64)
                for s0 in range(delay, n, delay):
                    s1 = min(n, s0 + delay)
                    ln = s1 - s0
                    yc[s0:s1] = x[s0 - delay : s0 - delay + ln] + fb * yc[s0 - delay : s0 - delay + ln]
                comb_sum += yc
            comb_sum /= float(len(ch_combs[ch]))

            # Gentle damping lowpass inside reverb tail for warm lush decay
            y = self._butter(comb_sum, cutoff=2800.0, btype="low", order=2)

            # 2 cascaded Schroeder allpass diffusers: y[i] = -g*x[i] + x[i - D] + g*y[i - D]
            for ap_d, g in allpasses:
                xa = y
                ya = -g * xa
                for s0 in range(ap_d, n, ap_d):
                    s1 = min(n, s0 + ap_d)
                    ln = s1 - s0
                    ya[s0:s1] += xa[s0 - ap_d : s0 - ap_d + ln] + g * ya[s0 - ap_d : s0 - ap_d + ln]
                y = ya

            out[:, ch] = x * (1.0 - wet) + y * wet
        return out

    def synth_eno_ambient_score(self) -> np.ndarray:
        """
        Brian Eno-style generative ambient score:
        5 detuned sine/warm voices modulated by ultra-slow incommensurate LFOs,
        morphing seamlessly across the 7-act emotional chord arc, passed through
        heavy algorithmic reverb — cutting to total silence at 97.5s-98.5s.
        """
        n = self.total_samples
        t = np.arange(n, dtype=np.float64) / self.sr

        # 5 voices: chord frequency table at act milestones (Hz)
        # Voice 0: deep root | Voice 1: fifth/tritone | Voice 2: third (m/M) | Voice 3: 7th/9th | Voice 4: high shimmer
        chord_knots = [
            (0.0,   [73.42, 110.00, 174.61, 261.63, 329.63]),  # Act I:  D minor 9 (melancholic night)
            (14.5,  [73.42, 110.00, 174.61, 261.63, 329.63]),
            (19.0,  [73.42, 110.00, 185.00, 277.18, 329.63]),  # Act II: D Major 9 (warm sunrise bloom!)
            (32.0,  [73.42, 110.00, 185.00, 293.66, 369.99]),  # Act III: Bright midday D major / shimmer
            (51.0,  [73.42, 110.00, 174.61, 246.94, 329.63]),  # Act IV: D Dorian mystery (camouflage)
            (69.5,  [73.42, 110.00, 174.61, 246.94, 311.13]),
            (74.0,  [73.42,  77.78, 103.83, 155.56, 207.65]),  # Act V:  Dissonant tritone/b2 cluster (Viper thermal!)
            (89.5,  [73.42,  77.78, 103.83, 155.56, 207.65]),
            (92.0,  [73.42, 110.00, 174.61, 220.00, 349.23]),  # Act VI: Dramatic Golden Hour minor swell
            (110.0, [73.42, 110.00, 174.61, 261.63, 329.63]),  # Act VII: Return to Act I Genesis D minor 9
            (120.0, [73.42, 110.00, 174.61, 261.63, 329.63]),
        ]
        knot_t = np.array([k[0] for k in chord_knots], dtype=np.float64)
        knot_f = np.array([k[1] for k in chord_knots], dtype=np.float64)

        # Ultra-slow asynchronous Eno LFO periods (seconds) & gentle stereo pans
        lfo_periods = [19.3, 13.7, 23.1, 17.3, 29.7]
        detunes = [-0.32, +0.41, -0.55, +0.62, -0.78]
        voice_pans = [0.0, -0.35, +0.38, -0.58, +0.60]
        voice_gains = [0.36, 0.24, 0.22, 0.16, 0.13]

        dry = np.zeros((n, 2), dtype=np.float64)
        # Aggression/grit mask during Act V Thermal Viper (74s -> 89.5s)
        viper_agg = np.clip((t - 73.5) / 1.8, 0.0, 1.0) * np.clip((90.5 - t) / 1.5, 0.0, 1.0)

        for v in range(5):
            f_t = np.interp(t, knot_t, knot_f[:, v])
            # Slow detuned breathing drift
            f_a = f_t + detunes[v] * np.sin(2.0 * np.pi * t / (lfo_periods[v] * 1.3))
            f_b = f_t - detunes[v] * 0.85 * np.cos(2.0 * np.pi * t / (lfo_periods[v] * 0.9))
            ph_a = 2.0 * np.pi * np.cumsum(f_a) / self.sr
            ph_b = 2.0 * np.pi * np.cumsum(f_b) / self.sr

            # Pure warm detuned sine pair + menacing FM/odd-harmonic grit ONLY in Act V
            wave = 0.58 * np.sin(ph_a) + 0.42 * np.sin(ph_b)
            if v in (0, 1, 2):
                grit = np.tanh(wave * 3.4 + 0.85 * np.sin(ph_a * 2.0 + 2.2 * np.sin(ph_b * 1.5)))
                wave = wave * (1.0 - 0.58 * viper_agg) + grit * (0.58 * viper_agg)

            # Overlapping ultra-slow LFO amplitude envelope
            lfo = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(2.0 * np.pi * t / lfo_periods[v] + v * 1.37))
            sig = wave * lfo * voice_gains[v]

            theta = (voice_pans[v] + 1.0) * (math.pi * 0.25)
            dry[:, 0] += sig * math.cos(theta)
            dry[:, 1] += sig * math.sin(theta)

        # Action-responsive macro swell across the 120s score
        macro = (
            0.55
            + 0.25 * np.clip((t - 15.0) / 10.0, 0.0, 1.0)  # Swell in Act II sunrise
            + 0.22 * viper_agg                              # Intense in Act V thermal
            + 0.28 * np.clip((t - 90.0) / 5.0, 0.0, 1.0) * np.clip((98.0 - t) / 1.0, 0.0, 1.0)  # Act VI pre-dive swell
        )
        # Gentle resolve & fade during Act VII Dust to Dust
        macro *= np.where(t > 109.0, 0.65 * np.clip((119.4 - t) / 10.4, 0.0, 1.0) + 0.0, 1.0)
        dry *= macro[:, None]

        # Heavy Schroeder algorithmic reverb applied BEFORE the hard 1-second silence cut
        # so that 97.50s -> 98.50s is genuinely TOTAL SILENCE as required!
        reverbed = self._algorithmic_reverb(dry, wet=0.62)

        # Clean dramatic drone cut-out during Act VI total silence [97.50s - 98.50s]
        cut_down = np.clip((97.48 - t) / 0.14, 0.0, 1.0)
        cut_up = np.clip((t - 101.8) / 3.0, 0.0, 1.0)
        gate = np.where(t < 97.48, 1.0, np.where(t < 101.8, 0.0, cut_up))
        gate = np.where((t >= 97.34) & (t < 97.48), cut_down, gate)
        return reverbed * gate[:, None] * 1.18

    def mix_binaural(
        self,
        master: np.ndarray,
        mono: np.ndarray,
        start_time: float,
        screen_x: float | np.ndarray,
        gain: float = 1.0,
    ) -> None:
        """
        Binaurally spatialize a mono foley clip using on-screen X coordinate:
        maps X in [0, WIDTH] -> stereo level pan + subtle Interaural Time
        Difference (ITD, up to +/- 0.65 ms delay on the far ear).
        """
        start_idx = int(round(start_time * self.sr))
        if start_idx >= self.total_samples:
            return
        clip = mono * gain
        if start_idx < 0:
            clip = clip[-start_idx:]
            start_idx = 0
        length = min(len(clip), self.total_samples - start_idx)
        if length <= 0:
            return
        clip = clip[:length]

        if isinstance(screen_x, np.ndarray):
            sx = np.interp(
                np.linspace(0, 1, length), np.linspace(0, 1, len(screen_x)), screen_x
            )
            pan = np.clip((sx / float(WIDTH)) * 2.0 - 1.0, -0.95, 0.95)
            theta = (pan + 1.0) * (math.pi * 0.25)
            master[start_idx : start_idx + length, 0] += clip * np.cos(theta)
            master[start_idx : start_idx + length, 1] += clip * np.sin(theta)
        else:
            pan_f = clamp((float(screen_x) / float(WIDTH)) * 2.0 - 1.0, -0.95, 0.95)
            theta = (pan_f + 1.0) * (math.pi * 0.25)
            # Interaural Time Difference (ITD): ~28 samples (~0.64ms) max ear lag
            itd = int(round(abs(pan_f) * 28.0))
            left = clip * math.cos(theta)
            right = clip * math.sin(theta)
            if pan_f > 0 and itd > 0 and itd < length:
                left = np.concatenate((np.zeros(itd), left[:-itd]))
            elif pan_f < 0 and itd > 0 and itd < length:
                right = np.concatenate((np.zeros(itd), right[:-itd]))
            master[start_idx : start_idx + length, 0] += left
            master[start_idx : start_idx + length, 1] += right

    # -------------------------------------------------------------------------
    # Individual Procedural Foley Synthesizers
    # -------------------------------------------------------------------------

    def synth_water_drop(self, pitch_hz: float = 1400.0) -> np.ndarray:
        """Short sine ping with rapid downward frequency sweep (water drop)."""
        dur = 0.13
        n = int(round(dur * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        # Classic resonant bubble ping: rapid downward frequency sweep
        freq = pitch_hz * (0.42 + 0.58 * np.exp(-t * 36.0))
        ph = 2.0 * np.pi * np.cumsum(freq) / self.sr
        env = np.exp(-t * 28.0) * (1.0 - np.exp(-t * 480.0))
        return self._fade(np.sin(ph) * env * 0.55, self.sr, 0.001, 0.015)

    def synth_beetle_click(self, freq_hz: float = 2400.0, crunch: bool = False) -> np.ndarray:
        """Act III Mechanical, clicky insect foley via rapid square-wave bursts."""
        dur = 0.11 if crunch else 0.055
        n = int(round(dur * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        if crunch:
            f = freq_hz * (1.0 - 0.35 * (t / dur)) + 320.0 * np.sin(2.0 * np.pi * 65.0 * t)
            sq = signal.square(2.0 * np.pi * np.cumsum(f) / self.sr)
            sq2 = signal.square(2.0 * np.pi * np.cumsum(f * 0.53) / self.sr)
            env = np.exp(-t * 32.0) * (0.55 + 0.45 * np.maximum(0.0, np.sin(2.0 * np.pi * 34.0 * t)))
            sig = self._butter((0.6 * sq + 0.4 * sq2) * env, (750.0, 5800.0), "band", 3)
            return self._fade(sig * 0.55, self.sr, 0.001, 0.01)
        else:
            f = freq_hz * np.exp(-t * 18.0)
            sq = signal.square(2.0 * np.pi * np.cumsum(f) / self.sr)
            env = np.exp(-t * 85.0)
            sig = self._butter(sq * env, (1100.0, 6200.0), "band", 3)
            return self._fade(sig * 0.42, self.sr, 0.0008, 0.008)

    def synth_chameleon_throat(self, duration: float = 2.4) -> np.ndarray:
        """Act IV Strange, warbling FM-synth reptile throat vocalization."""
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        u = t / duration
        warble = 26.0 * np.sin(2.0 * np.pi * (6.5 + 3.0 * u) * t)
        fc = 148.0 + 45.0 * np.sin(2.0 * np.pi * 1.4 * t) + warble
        fm1 = 42.0 + 18.0 * np.sin(2.0 * np.pi * 2.7 * t)
        fm2 = fc * 1.49
        ph_c = 2.0 * np.pi * np.cumsum(fc) / self.sr
        ph_m1 = 2.0 * np.pi * np.cumsum(fm1) / self.sr
        ph_m2 = 2.0 * np.pi * np.cumsum(fm2) / self.sr
        sig = np.sin(ph_c + 4.8 * np.sin(ph_m1) + 2.1 * np.sin(ph_m2))
        sig = self._butter(sig, (110.0, 1650.0), "band", 3)
        env = (np.sin(np.pi * u) ** 0.7) * (0.65 + 0.35 * np.sin(2.0 * np.pi * 7.5 * t))
        return self._fade(sig * env * 0.68, self.sr, 0.04, 0.08)

    def synth_tongue_whip_crack(self, duration: float = 0.42) -> np.ndarray:
        """Act IV Massive bass-heavy 'whip' crack for the ballistic chameleon tongue."""
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        # Deep sub-bass body punch
        f_sub = 145.0 * np.exp(-t * 22.0) + 28.0
        sub = np.sin(2.0 * np.pi * np.cumsum(f_sub) / self.sr) * np.exp(-t * 7.5) * 1.15
        # Supersonic high-transient whip snap at t = 0.03s
        dt = np.maximum(0.0, t - 0.028)
        crack_noise = self._butter(self.rng.standard_normal(n), (1200.0, 9500.0), "band", 3)
        crack = np.where(t >= 0.028, crack_noise * np.exp(-dt * 75.0) * 1.6, 0.0)
        # Elastic high-to-low whoosh zip
        f_zip = 2600.0 * np.exp(-t * 32.0) + 160.0
        zipper = np.sin(2.0 * np.pi * np.cumsum(f_zip) / self.sr) * np.exp(-t * 24.0) * 0.65
        return self._fade(np.tanh((sub + crack + zipper) * 1.15) * 0.95, self.sr, 0.001, 0.04)

    def synth_viper_hiss(self, duration: float = 12.5) -> np.ndarray:
        """Act V Terrifying high-frequency sibilant viper hiss synth."""
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        white = self.rng.standard_normal(n)
        hp = self._butter(white, 4300.0, "high", 5)
        sib1 = self._butter(white, (5400.0, 8600.0), "band", 3)
        # Metallic predatory harmonic whistle layered softly into hiss
        whistle = np.sin(2.0 * np.pi * (6200.0 + 180.0 * np.sin(2.0 * np.pi * 2.2 * t)) * t) * 0.05
        # Pulsing aggressive breath bursts peaking near strike (t ~ 6.2s inside clip)
        u = t / duration
        swell = 0.35 + 0.65 * np.exp(-((u - 0.56) / 0.25) ** 2)
        breath = 0.62 + 0.38 * np.sin(2.0 * np.pi * 1.35 * t)
        sig = (hp * 0.62 + sib1 * 0.68 + whistle) * swell * breath * 0.52
        return self._fade(sig, self.sr, 0.35, 0.55)

    def synth_eagle_impact_explosion(self, duration: float = 2.6) -> np.ndarray:
        """Act VI Massive white-noise impact explosion + seismic shockwave."""
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        white = self.rng.standard_normal(n)
        # Full-spectrum white-noise detonation + booming low-pass shockwave
        blast_hi = white * np.exp(-t * 5.2) * 0.95
        blast_lo = self._butter(white, 260.0, "low", 4) * np.exp(-t * 2.2) * 1.55
        sub_boom = np.sin(2.0 * np.pi * np.cumsum(92.0 * np.exp(-t * 9.0) + 26.0) / self.sr)
        sub_boom *= np.exp(-t * 3.5) * 1.25
        # Fluttering feather-burst air swoosh after impact
        feathers = self._butter(self.rng.standard_normal(n), (450.0, 2400.0), "band", 3)
        feathers *= np.exp(-t * 1.4) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 7.5 * t)) * 0.35
        out = np.tanh((blast_hi + blast_lo + sub_boom + feathers) * 0.95) * 0.98
        return self._fade(out, self.sr, 0.001, 0.25)

    def synth_eagle_screech(self, duration: float = 2.5) -> np.ndarray:
        """Act VI Piercing descending raptor screech via pitch-bent sawtooth wave."""
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        u = t / duration
        pitch = np.where(
            u < 0.07,
            2900.0 + 620.0 * (u / 0.07),
            3520.0 * ((1420.0 / 3520.0) ** ((u - 0.07) / 0.93)),
        )
        vib = (62.0 + 38.0 * u) * np.sin(2.0 * np.pi * (38.0 - 9.0 * u) * t)
        ph1 = 2.0 * np.pi * np.cumsum(pitch + vib) / self.sr
        ph2 = 2.0 * np.pi * np.cumsum((pitch + vib) * 0.993) / self.sr
        saw = 0.65 * signal.sawtooth(ph1) + 0.35 * signal.sawtooth(ph2)
        screech = self._butter(saw, (1100.0, 7200.0), "band", 4)
        env = np.where(u < 0.06, (u / 0.06) ** 0.5, np.exp(-1.75 * (u - 0.06)))
        return self._fade(screech * env * 0.95, self.sr, 0.008, 0.16)

    def synthesize_master_track(self) -> np.ndarray:
        """Compose the full 120.0-second stereo soundtrack (generative score + binaural foley)."""
        n = self.total_samples
        t = np.arange(n, dtype=np.float64) / self.sr
        master = np.zeros((n, 2), dtype=np.float64)

        # 1. Brian Eno-style Generative Ambient Drone Score
        master += self.synth_eno_ambient_score()

        # 2. Act I (0s-19s) & Act VII (111s-120s): Heavy Low-Pass Pink Noise Rain
        pink_l = self._butter(self._pink_noise(n), cutoff=1150.0, btype="low", order=4)
        pink_r = self._butter(self._pink_noise(n), cutoff=1150.0, btype="low", order=4)
        rain_env = (
            np.clip((19.5 - t) / 4.8, 0.0, 1.0) * np.clip(t / 0.9, 0.0, 1.0) * 0.24
            + np.clip((t - 111.2) / 2.2, 0.0, 1.0) * np.clip((119.5 - t) / 2.5, 0.0, 1.0) * 0.19
        )
        master[:, 0] += pink_l * rain_env
        master[:, 1] += pink_r * rain_env

        # Binaural Water Drop Synths synced to on-screen splash ripples in Act I & Act VII
        drop_events = (
            self.particles.get_water_drop_events(0.4, 15.8)
            + self.particles.get_water_drop_events(111.8, 118.2)
        )
        for idx, (dt, sx) in enumerate(drop_events):
            pitch = 1050.0 + (idx * 211 % 1050)
            gain = 0.62 if dt < 20.0 else 0.48
            self.mix_binaural(master, self.synth_water_drop(pitch), dt, sx, gain=gain)

        # 3. Act III (30s-50s): Binaural Mechanical Clicky Beetle Foley
        # Climbing footsteps (30.5s -> 41.5s, beetle climbs x: 340 -> 510)
        step_t = 30.6
        step_i = 0
        while step_t < 41.6:
            u = (step_t - 30.6) / 11.0
            bx = lerp(340.0, 510.0, u)
            freq = 1950.0 + (step_i % 4) * 380.0
            self.mix_binaural(master, self.synth_beetle_click(freq, crunch=False), step_t, bx, gain=0.92)
            step_t += 0.24 + 0.04 * (step_i % 3)
            step_i += 1
        # Eating fern crunches (41.8s -> 49.4s, beetle at x ~ 510)
        for k, ct in enumerate(np.arange(42.0, 49.2, 0.52)):
            self.mix_binaural(master, self.synth_beetle_click(1750.0 + (k % 3) * 260.0, crunch=True), float(ct), 510.0, gain=0.98)

        # 4. Act IV (50s-70s): Chameleon FM Throat Warble + Bass-Heavy Whip Crack Tongue
        self.mix_binaural(master, self.synth_chameleon_throat(2.5), 62.6, screen_x=910.0, gain=0.82)
        # Tongue whip sweeps rapidly across stereo field from Chameleon (right, 860) -> Beetle (left, 230)!
        whip = self.synth_tongue_whip_crack(0.45)
        pan_sweep = np.linspace(860.0, 230.0, len(whip))
        self.mix_binaural(master, whip, 66.65, screen_x=pan_sweep, gain=1.08)

        # 5. Act V (70s-90s): Terrifying High-Frequency Viper Hiss + Strike Thud
        hiss = self.synth_viper_hiss(13.2)
        viper_x_sweep = np.linspace(980.0, 620.0, len(hiss))
        self.mix_binaural(master, hiss, 75.2, screen_x=viper_x_sweep, gain=0.92)
        # Constrictor strike thud at 81.8s (centered on chameleon/viper at x ~ 740)
        strike_thud = self._fade(
            np.sin(2.0 * np.pi * np.cumsum(115.0 * np.exp(-np.arange(int(0.4 * self.sr)) / self.sr * 16.0) + 32.0) / self.sr)
            * np.exp(-np.arange(int(0.4 * self.sr)) / self.sr * 9.0),
            self.sr,
            0.002,
            0.04,
        )
        self.mix_binaural(master, strike_thud, 81.75, screen_x=740.0, gain=0.92)

        # 6. Act VI (90s-110s): Pre-Dive Wind -> 1.0s TOTAL SILENCE -> Explosion & Eagle Screech!
        dive_wind_n = int(4.6 * self.sr)
        dt_w = np.arange(dive_wind_n, dtype=np.float64) / self.sr
        dive_wind = self._butter(self.rng.standard_normal(dive_wind_n), (220.0, 1800.0), "band", 3)
        dive_wind *= np.clip(dt_w / 3.8, 0.0, 1.0) ** 1.6 * np.clip((4.55 - dt_w) / 0.12, 0.0, 1.0) * 0.45
        self.mix_binaural(master, dive_wind, 92.8, screen_x=640.0, gain=0.75)

        # Enforce ABSOLUTE 1.0-SECOND TOTAL SILENCE [97.50s - 98.50s]!
        s_sil0 = int(round(97.50 * self.sr))
        s_sil1 = int(round(98.50 * self.sr))
        fade_in_sil = int(0.06 * self.sr)
        master[s_sil0 - fade_in_sil : s_sil0] *= np.linspace(1.0, 0.0, fade_in_sil)[:, None]
        master[s_sil0:s_sil1] = 0.0

        # Massive White-Noise Explosion right at impact (98.50s)
        self.mix_binaural(master, self.synth_eagle_impact_explosion(2.6), 98.50, screen_x=680.0, gain=1.15)
        # Piercing Descending Eagle Screech sweeping left-to-right across the stereo field!
        screech = self.synth_eagle_screech(2.6)
        eagle_pan = np.linspace(480.0, 960.0, len(screech))
        self.mix_binaural(master, screech, 98.75, screen_x=eagle_pan, gain=0.92)

        # Master soft-limiting, normalization & clean fade-in / fade-to-silence
        master = np.tanh(master * 0.92)
        peak = float(np.max(np.abs(master)))
        if peak > 1e-6:
            master *= 0.92 / peak

        # Re-guarantee exact silence window [97.52s - 98.48s] and final fade-to-silence
        master[int(97.52 * self.sr) : int(98.48 * self.sr)] = 0.0
        fi = int(0.35 * self.sr)
        fo = int(1.80 * self.sr)
        master[:fi] *= np.linspace(0.0, 1.0, fi)[:, None]
        master[-fo:] *= (np.linspace(1.0, 0.0, fo) ** 2.0)[:, None]
        master[-int(0.20 * self.sr) :] = 0.0
        return master

    def export_wav(self, wav_path: str) -> str:
        stereo = self.synthesize_master_track()
        pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(wav_path, self.sr, pcm16)
        return wav_path


# =============================================================================
# MODULE 5: Cinematographer (7-Act Choreography, IK, Fractals & Multiprocessing)
# =============================================================================


class Cinematographer:
    """
    Master 120-second 7-Act director orchestrating SDF creatures, L-system
    bioluminescent mycelium, Fibonacci unrolling alien fern, 2-bone IK beetle,
    Voronoi camouflage chameleon, Thermal Vision viper, Golden Hour eagle dive,
    camera pan/shake, and parallel multiprocessing chunk rendering.
    """

    def __init__(self, width: int = WIDTH, height: int = HEIGHT, fps: int = FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.sdf = SDF_Renderer()
        self.lighting = Lighting_Engine(width, height)
        self.particles = Particle_System(width, height)

        # Precompute L-system mycelium grammars for Act I & Act VII
        self.mycelium_a = generate_lsystem("F", {"F": "F[+F]F[-F]F"}, iterations=3)
        self.mycelium_b = generate_lsystem("F", {"F": "F[-F]F[+F]"}, iterations=3)

    # -------------------------------------------------------------------------
    # Camera Rig: Panning & Massive Impact Shake
    # -------------------------------------------------------------------------

    def _camera_state(self, t: float) -> Tuple[float, float]:
        """Return world camera offset (cam_x, cam_y) including Act IV pan & Act VI shake."""
        # Act IV smooth pan right (50.0s -> 56.5s) to reveal the camouflaged chameleon!
        if t < 50.0:
            cam_x = 0.0
        elif t < 56.5:
            cam_x = 340.0 * ease_in_out_cubic((t - 50.0) / 6.5)
        elif t < 110.0:
            cam_x = 340.0
        elif t < 113.5:
            # Smooth settle back toward dark soil center in Act VII
            cam_x = 340.0 * (1.0 - ease_in_out_cubic((t - 110.0) / 3.5))
        else:
            cam_x = 0.0

        cam_y = 0.0
        # Act VI Massive Camera Shake on Eagle Impact (98.50s -> 101.6s)
        if 98.50 <= t <= 101.6:
            dt = t - 98.50
            amp = 46.0 * math.exp(-dt * 2.1)
            cam_x += amp * (math.sin(dt * 68.0) + 0.5 * math.cos(dt * 112.0))
            cam_y += amp * 0.85 * (math.cos(dt * 74.0) + 0.45 * math.sin(dt * 126.0))
        return cam_x, cam_y

    # -------------------------------------------------------------------------
    # Act I & VII: Bioluminescent Cyan L-System Mycelium
    # -------------------------------------------------------------------------

    def _draw_mycelium(
        self,
        canvas: np.ndarray,
        t: float,
        cam_x: float,
        origin_list: Sequence[Tuple[float, float, float, float]],
        growth: float,
        glow_strength: float,
    ) -> None:
        """Render glowing bioluminescent cyan L-system fungal hyphae networks."""
        if growth <= 0.01 or glow_strength <= 0.01:
            return
        buf = np.zeros_like(canvas)
        for idx, (ox, oy, base_len, start_ang_deg) in enumerate(origin_list):
            lstr = self.mycelium_a if idx % 2 == 0 else self.mycelium_b
            x, y = ox - cam_x, oy
            ang = math.radians(start_ang_deg)
            stack: List[Tuple[float, float, float, int]] = []
            depth = 0
            # Progressive character budget so hyphae physically grow outward over time
            max_chars = int(len(lstr) * clamp(growth))
            for ch in lstr[:max_chars]:
                if ch == "F":
                    seg = base_len * (0.76 ** depth) * (0.35 + 0.65 * growth)
                    sway = 0.05 * math.sin(t * 1.4 + idx + depth)
                    nx = x + seg * math.cos(ang + sway)
                    ny = y + seg * math.sin(ang + sway)
                    inten = (0.95 - 0.14 * depth) * glow_strength
                    col = (1.0 * inten, 0.94 * inten, 0.16 * inten)  # Bioluminescent cyan BGR
                    cv2.line(buf, (int(x), int(y)), (int(nx), int(ny)), col, max(1, 3 - depth), cv2.LINE_AA)
                    x, y = nx, ny
                elif ch == "+":
                    ang += math.radians(24.0 + 4.0 * (idx % 3))
                elif ch == "-":
                    ang -= math.radians(24.0 + 3.0 * ((idx + 1) % 3))
                elif ch == "[":
                    stack.append((x, y, ang, depth))
                    depth += 1
                elif ch == "]":
                    # Glowing cyan bioluminescent fungal spore bulb at branch tips
                    if depth >= 1:
                        pulse = 0.72 + 0.28 * math.sin(t * 3.5 + x * 0.05)
                        r_tip = max(2, int(4.5 * growth))
                        bcol = (1.0 * glow_strength * pulse, 0.98 * glow_strength * pulse, 0.35 * glow_strength * pulse)
                        cv2.circle(buf, (int(x), int(y)), r_tip, bcol, -1, cv2.LINE_AA)
                    if stack:
                        x, y, ang, depth = stack.pop()

        # Lush additive Gaussian bioluminescent bloom via fast half-res pass!
        half = cv2.resize(buf, (self.width // 2, self.height // 2), interpolation=cv2.INTER_AREA)
        bloom_tight = cv2.GaussianBlur(half, (0, 0), 2.5)
        bloom_wide = cv2.GaussianBlur(cv2.resize(half, (320, 180), interpolation=cv2.INTER_AREA), (0, 0), 5.0)
        canvas += (
            buf * np.float32(0.95)
            + cv2.resize(bloom_tight, (self.width, self.height)) * np.float32(0.85)
            + cv2.resize(bloom_wide, (self.width, self.height)) * np.float32(0.95)
        )

    # -------------------------------------------------------------------------
    # Act II - IV: Fractal Alien Fern with Fibonacci Spiral Unrolling
    # -------------------------------------------------------------------------

    def _fern_stem_point(self, s: float, unroll: float, t: float) -> Tuple[float, float, float]:
        """
        Compute point (world_x, world_y, tangent_angle) along the primary
        Fibonacci-unrolling alien fern frond for arc fraction s in [0, 1].
        """
        phi = 1.61803398875
        base_x, base_y = 335.0, 610.0
        n_steps = 46
        target_i = s * n_steps
        x, y = base_x, base_y
        # Starts aiming upward-right, curving gracefully into a tight Fibonacci spiral at tip
        ang = -math.pi * 0.44 + 0.03 * math.sin(t * 1.1)
        seg_len = 9.4 * (0.20 + 0.80 * smoothstep(unroll * 1.25))
        # Fiddlehead coil curvature decays smoothly as `unroll` approaches 1.0
        coil_strength = (1.0 - ease_out_cubic(unroll)) * 0.34
        for i in range(int(target_i) + 1):
            frac = min(1.0, target_i - i) if i == int(target_i) else 1.0
            u_i = i / float(n_steps)
            # Curvature follows golden-ratio Fibonacci progression toward tip
            d_ang = 0.016 + coil_strength * (u_i ** (1.0 / phi)) * 1.35
            ang += d_ang * frac
            x += seg_len * frac * math.cos(ang)
            y += seg_len * frac * math.sin(ang)
        return x, y, ang

    def _draw_alien_fern(
        self,
        canvas: np.ndarray,
        t: float,
        light: LightingState,
        cam_x: float,
        eaten_amount: float = 0.0,
    ) -> None:
        """Render the fractal alien fern unrolling via Fibonacci spiral math with SSS glow."""
        if t < 15.2 or t > 74.0:
            return
        unroll = clamp((t - 15.2) / 13.5)
        if unroll <= 0.01:
            return

        phi = 1.61803398875
        golden_angle = math.radians(137.507764)  # Fibonacci phyllotaxis angle!

        # Draw 3 fronds (hero center frond + 2 flanking companion fronds)
        fronds = [
            (0.0, 0.0, 1.00, 1.0, 0.0),      # Hero climbing frond
            (-65.0, 12.0, 0.68, -0.85, 0.8), # Left companion frond
            (58.0, 16.0, 0.62, 0.72, 1.4),   # Right companion frond
        ]

        glow_buf = np.zeros_like(canvas)
        sun_boost = 0.45 + 0.75 * light.sun_intensity

        for f_dx, f_dy, f_scale, curve_dir, delay in fronds:
            u_f = clamp((t - (15.2 + delay)) / 12.5)
            if u_f <= 0.02:
                continue
            n_pts = 44
            stem_pts = []
            stem_angs = []
            x, y = 335.0 + f_dx - cam_x, 610.0 + f_dy
            ang = -math.pi * 0.44 + (0.0 if curve_dir > 0 else -0.35) + 0.025 * math.sin(t * 1.2 + delay)
            seg_len = 9.4 * f_scale * (0.20 + 0.80 * smoothstep(u_f * 1.25))
            coil_str = (1.0 - ease_out_cubic(u_f)) * 0.34

            for i in range(n_pts):
                u_i = i / float(n_pts - 1)
                stem_pts.append((x, y))
                stem_angs.append(ang)
                d_ang = (0.016 + coil_str * (u_i ** (1.0 / phi)) * 1.35) * curve_dir
                ang += d_ang
                x += seg_len * math.cos(ang)
                y += seg_len * math.sin(ang)

            # Draw fractal recursive pinnae (leaflets) unfurling from base to coiled tip
            for i in range(4, n_pts - 2, 2):
                u_i = i / float(n_pts - 1)
                # Beetles eats upper pinnae of the hero frond in Act III!
                if f_scale > 0.9 and u_i > 0.72 and eaten_amount > 0.0:
                    if (u_i - 0.72) / 0.28 < eaten_amount:
                        continue
                # Each pinna unfurls once the local stem section straightens
                leaf_open = smoothstep((u_f * 1.35 - u_i * 0.75) / 0.35)
                if leaf_open <= 0.03:
                    continue
                px, py = stem_pts[i]
                if px < -120 or px > self.width + 120:
                    continue
                base_a = stem_angs[i]
                pinna_len = (58.0 * f_scale) * math.sin(math.pi * (u_i ** 0.75)) * leaf_open
                if pinna_len < 4.0:
                    continue

                # Alternating Fibonacci phyllotaxis side leaflets + level-2 fractal sub-leaflets
                for side in (-1.0, 1.0):
                    # Subtle Fibonacci rotation offset + unrolling sweep
                    la = base_a + side * (1.12 - 0.35 * (1.0 - leaf_open)) + 0.08 * math.sin(golden_angle * i)
                    lx = px + pinna_len * math.cos(la)
                    ly = py + pinna_len * math.sin(la)
                    # Emerald-chartreuse backlit translucent frond palette
                    leaf_col = (
                        0.14 * sun_boost,
                        (0.54 + 0.28 * u_i) * sun_boost,
                        (0.24 + 0.30 * u_i) * sun_boost,
                    )
                    sss_tip = (
                        0.18 * sun_boost,
                        0.82 * sun_boost,
                        0.58 * sun_boost,
                    )
                    cv2.line(canvas, (int(px), int(py)), (int(lx), int(ly)), leaf_col, max(1, int(3.0 * f_scale)), cv2.LINE_AA)
                    # Level-2 fractal sub-pinnae (recursive self-similarity!)
                    for k in range(1, 5):
                        s_sub = k / 5.0
                        sx = lerp(px, lx, s_sub)
                        sy = lerp(py, ly, s_sub)
                        sub_l = pinna_len * 0.34 * (1.0 - 0.65 * s_sub)
                        for s2 in (-1.0, 1.0):
                            sa2 = la + s2 * 0.68
                            ex, ey = sx + sub_l * math.cos(sa2), sy + sub_l * math.sin(sa2)
                            cv2.line(canvas, (int(sx), int(sy)), (int(ex), int(ey)), leaf_col, 1, cv2.LINE_AA)
                    # Glowing alien bioluminescent-gold spore node at pinna tip
                    cv2.circle(glow_buf, (int(lx), int(ly)), max(1, int(2.6 * f_scale)), sss_tip, -1, cv2.LINE_AA)

            # Main stem rachis
            pts_i = np.array(stem_pts, dtype=np.int32)
            stem_col = (0.10 * sun_boost, 0.44 * sun_boost, 0.18 * sun_boost)
            for i in range(len(pts_i) - 1):
                th = max(2, int((7.0 - 4.5 * (i / n_pts)) * f_scale))
                cv2.line(canvas, tuple(pts_i[i]), tuple(pts_i[i + 1]), stem_col, th, cv2.LINE_AA)

        half_g = cv2.resize(glow_buf, (self.width // 2, self.height // 2), interpolation=cv2.INTER_AREA)
        canvas += cv2.resize(cv2.GaussianBlur(half_g, (0, 0), 2.2), (self.width, self.height)) * np.float32(0.70)

    # -------------------------------------------------------------------------
    # Act III - IV: SDF Iridescent Rhinoceros Beetle with 6-Leg Analytical IK
    # -------------------------------------------------------------------------

    def _beetle_world_pos(self, t: float) -> Optional[Tuple[float, float, float, float]]:
        """Return (world_x, world_y, body_angle, eat_phase) for the Rhinoceros Beetle."""
        # Enters at 30.0s, gets snatched & disappears inside chameleon mouth at 67.25s
        if t < 30.0 or t >= 67.25:
            return None

        # Climb along the hero alien fern curve from s=0.05 (30s) -> s=0.74 (41.5s)
        if t < 41.5:
            s = lerp(0.06, 0.74, ease_in_out_cubic((t - 30.0) / 11.5))
            wx, wy, tang = self._fern_stem_point(s, 1.0, t)
            return wx, wy, tang, 0.0

        # Eating upper fern frond (41.5s -> 66.85s)
        wx, wy, tang = self._fern_stem_point(0.74, 1.0, t)
        eat_ph = math.sin((t - 41.5) * 10.5) if t < 50.5 else 0.25 * math.sin(t * 3.0)

        # Yanked rapidly right into chameleon's mouth by ballistic tongue (66.85s -> 67.25s)!
        if t >= 66.85:
            u = ease_in_quad((t - 66.85) / 0.40)
            cham_mouth_x, cham_mouth_y = 1170.0, 350.0
            wx = lerp(wx, cham_mouth_x, u)
            wy = lerp(wy, cham_mouth_y, u)
            tang += u * 1.4
        return wx, wy, tang, eat_ph

    def _draw_sdf_beetle(
        self,
        canvas: np.ndarray,
        heat_map: Optional[np.ndarray],
        t: float,
        light: LightingState,
        cam_x: float,
    ) -> None:
        """
        Render the SDF Rhinoceros Beetle: smooth-min blended carapace, curved
        horn, 6 IK legs smoothly melted into the thorax, and view-angle
        emerald-to-violet Iridescence shader!
        """
        state = self._beetle_world_pos(t)
        if state is None:
            return
        wx, wy, ang, eat_ph = state
        cx, cy = wx - cam_x, wy
        if cx < -160 or cx > self.width + 160:
            return

        # Bounding box around beetle
        pad = 135
        x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
        x1, y1 = min(self.width, int(cx) + pad), min(self.height, int(cy) + pad)
        if x1 <= x0 + 8 or y1 <= y0 + 8:
            return

        X, Y = self.sdf.make_grid(x0, y0, x1, y1, scale=0.75)
        ca, sa = math.cos(ang), math.sin(ang)
        head_nod = 0.14 * eat_ph

        def to_sc(lx: float, ly: float) -> Tuple[float, float]:
            return (cx + lx * ca - ly * sa, cy + lx * sa + ly * ca)

        # 1. Core SDF Carapace (elytra abdomen + armored pronotum + head)
        ab_c = to_sc(-22.0, -2.0)
        th_c = to_sc(16.0, -4.0)
        hd_c = to_sc(44.0, 4.0 + 4.0 * eat_ph)

        d_ab = self.sdf.sd_ellipse(X, Y, ab_c[0], ab_c[1], 38.0, 26.0, ang - 0.08)
        d_th = self.sdf.sd_ellipse(X, Y, th_c[0], th_c[1], 24.0, 22.0, ang + 0.08)
        d_hd = self.sdf.sd_ellipse(X, Y, hd_c[0], hd_c[1], 14.0, 13.0, ang + head_nod)
        d_body = self.sdf.smin(d_ab, d_th, 10.0)
        d_body = self.sdf.smin(d_body, d_hd, 8.0)

        # 2. Iconic Sweeping Curved Rhinoceros Horn (tapered SDF polyline smoothly melted into head!)
        horn_pts = [
            to_sc(48.0, -1.0 + 3.0 * eat_ph),
            to_sc(72.0, -10.0 + 2.0 * eat_ph),
            to_sc(88.0, -32.0 + 1.0 * eat_ph),
            to_sc(82.0, -52.0),
        ]
        d_horn = self.sdf.sd_polyline_tapered(X, Y, horn_pts, [8.5, 6.2, 4.0, 2.0], blend_k=5.0)
        # Smaller secondary pronotum prong
        prong_a, prong_b = to_sc(24.0, -22.0), to_sc(42.0, -34.0)
        d_prong = self.sdf.sd_capsule(X, Y, prong_a[0], prong_a[1], prong_b[0], prong_b[1], 6.0, 2.2)
        d_total = self.sdf.smin(d_body, d_horn, 6.5)
        d_total = self.sdf.smin(d_total, d_prong, 5.5)

        # 3. 6 Articulated Legs solved with Analytical 2-Bone IK & melted via `smin` into thorax!
        gait_t = t * 5.2 if t < 41.5 else 0.0
        leg_specs = [
            # (hip_lx, hip_ly, rest_fx, rest_fy, tripod_phase, bend)
            (28.0, 12.0, 56.0, 44.0, 0.0, -1.0),    # Front near
            (6.0, 14.0, 14.0, 50.0, math.pi, -1.0), # Mid near
            (-18.0, 12.0, -38.0, 48.0, 0.0, 1.0),   # Hind near
            (30.0, -8.0, 58.0, 32.0, math.pi, -1.0),# Front far
            (8.0, -10.0, 18.0, 36.0, 0.0, -1.0),    # Mid far
            (-16.0, -10.0, -32.0, 36.0, math.pi, 1.0), # Hind far
        ]
        for h_lx, h_ly, f_lx, f_ly, ph, bend in leg_specs:
            swing = max(0.0, math.sin(gait_t + ph))
            stride = 14.0 * math.cos(gait_t + ph) if t < 41.5 else 0.0
            lift = -12.0 * swing if t < 41.5 else 0.0
            hip_pt = to_sc(h_lx, h_ly)
            target_pt = to_sc(f_lx + stride, f_ly + lift)
            knee_pt, foot_pt = solve_ik_2bone(hip_pt, target_pt, l1=28.0, l2=30.0, bend_sign=bend)
            d_fem = self.sdf.sd_capsule(X, Y, hip_pt[0], hip_pt[1], knee_pt[0], knee_pt[1], 5.2, 3.6)
            d_tib = self.sdf.sd_capsule(X, Y, knee_pt[0], knee_pt[1], foot_pt[0], foot_pt[1], 3.4, 1.8)
            d_leg = self.sdf.smin(d_fem, d_tib, 3.2)
            # Smoothly melt every IK leg directly into the thorax SDF!
            d_total = self.sdf.smin(d_total, d_leg, 5.0)

        # 4. Compute 3D normals + Iridescence shader (shifts emerald <-> violet with motion!)
        nx, ny, nz, depth = self.sdf.compute_normals_3d(d_total, max_depth=30.0)
        # Dynamic phase shift from beetle position + angle so carapace visibly shimmers as it climbs!
        irid_phase = t * 0.65 + (wx * 0.015) + ang * 1.8
        color_bgr = self.sdf.shade_iridescence(nx, ny, nz, light.light_dir, phase_shift=irid_phase)

        # Subtle sculpted wing-case seam crease
        seam = np.abs(self.sdf.sd_capsule(X, Y, to_sc(-50, -3)[0], to_sc(-50, -3)[1], to_sc(5, -5)[0], to_sc(5, -5)[1], 0.5))
        crease_dark = np.float32(1.0) - np.float32(0.38) * np.exp(-seam * seam / np.float32(6.0))
        color_bgr *= crease_dark[..., None]

        # Ambient occlusion + slight warm backlit rim
        color_bgr *= (np.float32(0.55) + np.float32(0.55) * light.sun_intensity)

        self.sdf.composite_sdf(
            canvas,
            heat_map,
            (x0, y0, x1, y1),
            d_total,
            color_bgr,
            heat_value=0.35,
            shadow_offset=(8, 12),
            shadow_alpha=0.32,
        )

    # -------------------------------------------------------------------------
    # Act IV - VI: Mossy Tree Branch & Voronoi Camouflage SDF Chameleon
    # -------------------------------------------------------------------------

    def _draw_tree_branch(
        self,
        canvas: np.ndarray,
        heat_map: Optional[np.ndarray],
        t: float,
        light: LightingState,
        cam_x: float,
    ) -> None:
        """Render the gnarled mossy jungle tree branch across the right scene."""
        if t < 48.0 or t > 113.5:
            return
        # Branch in world coords: extends from right edge (x=1720) leftward to x=860
        pts = [
            (1720.0 - cam_x, 425.0),
            (1420.0 - cam_x, 452.0),
            (1160.0 - cam_x, 466.0),
            (940.0 - cam_x, 488.0),
            (795.0 - cam_x, 515.0),
        ]
        radii = [58.0, 48.0, 40.0, 30.0, 16.0]
        x0 = max(0, int(min(p[0] for p in pts)) - 65)
        y0 = max(0, int(min(p[1] for p in pts)) - 75)
        x1 = min(self.width, int(max(p[0] for p in pts)) + 65)
        y1 = min(self.height, int(max(p[1] for p in pts)) + 75)
        if x1 <= x0 + 8 or y1 <= y0 + 8:
            return

        # Evaluate branch SDF at 0.5x for speed then upsample smoothly (organic bark!)
        X, Y = self.sdf.make_grid(x0, y0, x1, y1, scale=0.5)
        d = self.sdf.sd_polyline_tapered(X, Y, pts, radii, blend_k=14.0)
        # Small stub twig branching off underneath
        d_twig = self.sdf.sd_capsule(X, Y, 1260 - cam_x, 475, 1185 - cam_x, 545, 14.0, 5.0)
        d = self.sdf.smin(d, d_twig, 10.0)

        nx, ny, nz, _ = self.sdf.compute_normals_3d(d, max_depth=44.0)
        # Sample procedural bark texture in world space so it stays locked during camera pan!
        tex_h, tex_w = self.lighting.bark_tex.shape[:2]
        tx = (np.round(X + np.float32(cam_x)).astype(np.int32)) % tex_w
        ty = (np.round(Y).astype(np.int32)) % tex_h
        albedo = self.lighting.bark_tex[ty, tx]

        ndotl = np.clip(nx * light.light_dir[0] + ny * light.light_dir[1] + nz * light.light_dir[2], 0.0, 1.0)
        lit = albedo * (light.ambient_bgr * 0.75 + light.sun_color_bgr * (ndotl * light.sun_intensity * 0.95)[..., None])
        self.sdf.composite_sdf(
            canvas,
            heat_map,
            (x0, y0, x1, y1),
            d,
            lit,
            heat_value=0.19,  # Cool inanimate wood in thermal vision
            shadow_offset=(0, 18),
            shadow_alpha=0.25,
        )

    def _draw_sdf_chameleon(
        self,
        canvas: np.ndarray,
        heat_map: Optional[np.ndarray],
        t: float,
        light: LightingState,
        cam_x: float,
    ) -> None:
        """
        Render the SDF Panther Chameleon perched on the branch:
          - Seamless smooth-min SDF body, domed casque head, gripping legs & spiral tail
          - Animated Voronoi camouflage mask shifting bark -> vibrant red & blue
          - Fake backlit red/orange Subsurface Scattering (SSS)
          - Independently rotating geometric turret eyes
          - Motion-blurred ballistic tongue strike at 66.65s -> 67.25s!
        """
        if t < 49.0 or t > 90.8:
            return

        # Chameleon sits on the branch facing LEFT toward the beetle!
        # World anchor: (1250, 388) -> after pan (cam_x=340), screen_x = 910
        wx, wy = 1245.0, 388.0
        cx, cy = wx - cam_x, wy
        if cx < -240 or cx > self.width + 240:
            return

        # Throat warble inflation (62.5s - 65.8s) + mouth open for tongue (66.5s - 67.35s)
        throat_puff = 0.0
        if 62.5 <= t <= 65.8:
            u_th = (t - 62.5) / 3.3
            throat_puff = math.sin(u_th * math.pi) * (0.65 + 0.35 * math.sin(t * 14.0))
        mouth_open = 0.0
        if 66.45 <= t <= 67.40:
            mouth_open = math.sin(clamp((t - 66.45) / 0.95) * math.pi)

        # Slight tremble when viper drops & coils (79s - 90s)
        trem = 2.5 * math.sin(t * 24.0) if (79.0 <= t <= 90.0) else 0.0
        cx += trem

        x0, y0 = max(0, int(cx) - 185), max(0, int(cy) - 135)
        x1, y1 = min(self.width, int(cx) + 235), min(self.height, int(cy) + 145)
        if x1 <= x0 + 8 or y1 <= y0 + 8:
            return

        X, Y = self.sdf.make_grid(x0, y0, x1, y1, scale=0.60)

        # 1. Arched SDF Torso + Casque Helmet Head + Inflatable Throat Pouch
        d_body = self.sdf.sd_ellipse(X, Y, cx + 18, cy + 4, 88.0, 48.0, -0.08)
        d_back = self.sdf.sd_ellipse(X, Y, cx + 6, cy - 18, 72.0, 34.0, 0.06)
        d_head = self.sdf.sd_ellipse(X, Y, cx - 78, cy - 18, 44.0, 29.0, 0.14)
        # Signature tall triangular chameleon casque crest
        d_casque = self.sdf.sd_capsule(X, Y, cx - 54, cy - 30, cx - 36, cy - 72, 20.0, 6.0)
        # Lower jaw / inflatable warbling throat sac
        d_throat = self.sdf.sd_ellipse(
            X, Y, cx - 74, cy + 12 + 12.0 * throat_puff, 32.0 + 10.0 * throat_puff, 16.0 + 14.0 * throat_puff, 0.1
        )
        # Open jaw cutout during ballistic tongue strike
        if mouth_open > 0.05:
            d_jaw_cut = self.sdf.sd_capsule(X, Y, cx - 126, cy - 10, cx - 82, cy - 6, 12.0 * mouth_open, 5.0 * mouth_open)
            d_head = self.sdf.smax(d_head, -d_jaw_cut, 3.0)

        d_cham = self.sdf.smin(d_body, d_back, 16.0)
        d_cham = self.sdf.smin(d_cham, d_head, 14.0)
        d_cham = self.sdf.smin(d_cham, d_casque, 12.0)
        d_cham = self.sdf.smin(d_cham, d_throat, 12.0)

        # 2. Prehensile Logarithmic-Spiral Curled Tail (tapered SDF polyline!)
        tail_pts = []
        tail_r = []
        for k in range(13):
            u = k / 12.0
            ang_t = -0.35 + u * 4.55  # Coils gracefully underneath the branch
            rad_t = 76.0 * (1.0 - 0.68 * u)
            tx = cx + 96.0 + rad_t * math.cos(ang_t) * 0.85
            ty = cy + 44.0 + rad_t * math.sin(ang_t) * 0.72
            tail_pts.append((tx, ty))
            tail_r.append(lerp(22.0, 4.2, u))
        d_tail = self.sdf.sd_polyline_tapered(X, Y, tail_pts, tail_r, blend_k=8.0)
        d_cham = self.sdf.smin(d_cham, d_tail, 14.0)

        # 3. Gripping Foreleg & Hindleg (smoothly melted into torso via smin)
        d_f1 = self.sdf.sd_capsule(X, Y, cx - 32, cy + 28, cx - 44, cy + 72, 13.0, 9.5)
        d_f2 = self.sdf.sd_capsule(X, Y, cx - 44, cy + 72, cx - 32, cy + 92, 9.5, 8.5)
        d_h1 = self.sdf.sd_capsule(X, Y, cx + 62, cy + 24, cx + 76, cy + 64, 15.0, 10.0)
        d_h2 = self.sdf.sd_capsule(X, Y, cx + 76, cy + 64, cx + 56, cy + 84, 10.0, 8.5)
        d_cham = self.sdf.smin(d_cham, self.sdf.smin(d_f1, d_f2, 5.0), 10.0)
        d_cham = self.sdf.smin(d_cham, self.sdf.smin(d_h1, d_h2, 5.0), 10.0)

        # 4. Normals & Animated Voronoi Camouflage Reveal (Bark -> Vibrant Red & Blue!)
        nx, ny, nz, depth = self.sdf.compute_normals_3d(d_cham, max_depth=40.0)
        vor_d, vor_id = self.sdf.voronoi_field(X + np.float32(cam_x), Y, cell_size=22.0, seed=77)

        # Bark camouflage albedo sampled from identical world bark texture
        tex_h, tex_w = self.lighting.bark_tex.shape[:2]
        btx = (np.round(X + np.float32(cam_x)).astype(np.int32)) % tex_w
        bty = (np.round(Y).astype(np.int32)) % tex_h
        bark_col = self.lighting.bark_tex[bty, btx]

        # Vibrant Red & Blue Panther Chameleon striped chromatophore pattern!
        stripe_wave = 0.5 + 0.5 * np.sin((X - np.float32(cx)) * np.float32(0.095) + np.sin(Y * np.float32(0.05)) * np.float32(1.4))
        vivid_red_bgr = np.array([0.14, 0.22, 0.96], dtype=np.float32)
        vivid_blue_bgr = np.array([0.96, 0.52, 0.08], dtype=np.float32)
        vivid_teal_bgr = np.array([0.82, 0.86, 0.14], dtype=np.float32)
        sw = stripe_wave[..., None]
        cham_vivid = vivid_red_bgr * sw + vivid_blue_bgr * (1.0 - sw)
        # Subtle turquoise accent on Voronoi cell centers
        cham_vivid = cham_vivid * (1.0 - 0.28 * vor_id[..., None]) + vivid_teal_bgr * (0.28 * vor_id[..., None])

        # Animated Voronoi wave sweeping head-to-tail from t=54.5s -> 62.5s!
        reveal_prog = clamp((t - 54.5) / 7.5)
        # Spatial progression from head (x = cx - 120) toward tail (x = cx + 180)
        spatial_u = np.clip((X - np.float32(cx - 125.0)) / np.float32(310.0), 0.0, 1.0)
        # Voronoi cellular threshold makes individual polygonal skin cells flip color!
        cell_thresh = spatial_u * np.float32(0.75) + vor_id * np.float32(0.32) + vor_d * np.float32(0.15)
        vor_mask = smoothstep_arr(-0.08, 0.08, np.float32(reveal_prog * 1.25) - cell_thresh)
        # Bright iridescent cellular rim glow right along the revealing Voronoi wave front
        wave_edge = np.exp(-((np.float32(reveal_prog * 1.25) - cell_thresh) / np.float32(0.065)) ** 2)
        wave_edge *= np.float32(1.0 if (54.5 < t < 63.0) else 0.0)

        albedo = bark_col * (1.0 - vor_mask[..., None]) + cham_vivid * vor_mask[..., None]

        # Directional lighting + Fake Backlit Red/Orange Subsurface Scattering (SSS)!
        ndotl = np.clip(nx * light.light_dir[0] + ny * light.light_dir[1] + nz * light.light_dir[2], 0.0, 1.0)
        lit = albedo * (light.ambient_bgr * 0.72 + light.sun_color_bgr * (ndotl * light.sun_intensity * 0.95)[..., None])
        sss = self.sdf.shade_sss(
            d_cham, nx, ny, nz, light.light_dir,
            sss_color_bgr=(0.10, 0.34, 1.00),
            sss_strength=0.65 * (0.25 + 0.75 * reveal_prog),
            sss_depth=18.0,
        )
        lit += sss + wave_edge[..., None] * np.array([1.0, 0.82, 0.35], dtype=np.float32) * np.float32(0.75)

        # Warm crimson-orange living heat signature for Act V Thermal Heatmap Vision
        cham_heat = np.clip(np.float32(0.50) + depth * np.float32(0.28) + np.float32(0.03 * math.sin(t * 4.0)), 0.0, 0.80)

        self.sdf.composite_sdf(
            canvas,
            heat_map,
            (x0, y0, x1, y1),
            d_cham,
            lit,
            heat_value=cham_heat,
            shadow_offset=(4, 10),
            shadow_alpha=0.25,
        )

        # 5. Independently Rotating Geometric Turret Eyes (drawn cleanly atop head!)
        # Eye 1 (near large turret) and Eye 2 (far peeking ridge turret) rotate independently!
        eye_specs = [
            (cx - 86.0, cy - 22.0, 16.0, 0.0),   # Near turret eye
            (cx - 104.0, cy - 30.0, 11.5, 1.0),  # Far turret eye (peeking over snout)
        ]
        for ex, ey, er, is_far in eye_specs:
            if t < 60.5:
                # Independent asynchronous geometric scanning in opposite orbits!
                gaze_a = (1.6 * t + 0.5) if is_far < 0.5 else (-2.1 * t + 2.4)
                px = ex + (er * 0.42) * math.cos(gaze_a)
                py = ey + (er * 0.42) * math.sin(gaze_a)
            else:
                # Both eyes snap sharply LEFT to lock onto the beetle!
                lock_u = smoothstep((t - 60.5) / 0.8)
                gaze_a = lerp((1.6 * 60.5) if is_far < 0.5 else (-2.1 * 60.5), math.pi + 0.14, lock_u)
                px = ex + (er * 0.45) * math.cos(gaze_a)
                py = ey + (er * 0.45) * math.sin(gaze_a)

            # Geometric faceted concentric eye turret
            turret_col = (
                lerp(0.18, 0.20, reveal_prog),
                lerp(0.24, 0.72, reveal_prog),
                lerp(0.22, 0.92, reveal_prog),
            )
            cv2.circle(canvas, (int(ex), int(ey)), int(er), turret_col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(ex), int(ey)), int(er), (0.08, 0.12, 0.16), 2, cv2.LINE_AA)
            # Concentric geometric iris ring + dark tracking pupil
            cv2.circle(canvas, (int(px), int(py)), max(3, int(er * 0.44)), (0.18, 0.82, 0.98), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(px), int(py)), max(2, int(er * 0.24)), (0.04, 0.04, 0.05), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(px - 2), int(py - 2)), 2, (0.95, 0.98, 1.0), -1, cv2.LINE_AA)

        # 6. Ballistic Motion-Blurred Tongue Strike (66.62s -> 67.28s)!
        if 66.62 <= t <= 67.28:
            # Fast out (66.62->66.85), contact (66.85), snap back (66.85->67.28)
            if t <= 66.85:
                ext = ease_out_quad((t - 66.62) / 0.23)
            else:
                ext = 1.0 - ease_in_quad((t - 66.85) / 0.43)

            m_pt = (cx - 114.0, cy - 8.0)
            b_wx, b_wy, _ = self._fern_stem_point(0.74, 1.0, t)
            tgt = (b_wx - cam_x, b_wy)
            tip = (lerp(m_pt[0], tgt[0], ext), lerp(m_pt[1], tgt[1], ext))

            # Render multi-sample motion-blurred ballistic tongue line!
            t_buf = np.zeros_like(canvas)
            for s_k in range(5):
                sub_ext = clamp(ext - s_k * 0.045)
                sub_tip = (lerp(m_pt[0], tgt[0], sub_ext), lerp(m_pt[1], tgt[1], sub_ext))
                w_a = (1.0 - s_k * 0.17) * 0.55
                cv2.line(
                    t_buf,
                    (int(m_pt[0]), int(m_pt[1])),
                    (int(sub_tip[0]), int(sub_tip[1])),
                    (0.32 * w_a, 0.36 * w_a, 0.98 * w_a),
                    max(4, 12 - s_k * 2),
                    cv2.LINE_AA,
                )
            # Horizontal directional motion blur along strike axis
            t_blur = cv2.GaussianBlur(t_buf, (21, 3), 0)
            canvas += t_blur * np.float32(0.85)
            # Crisp glowing SSS sticky tongue bulb at tip
            cv2.line(canvas, (int(m_pt[0]), int(m_pt[1])), (int(tip[0]), int(tip[1])), (0.38, 0.42, 0.98), 8, cv2.LINE_AA)
            cv2.circle(canvas, (int(tip[0]), int(tip[1])), 16, (0.42, 0.48, 1.0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(tip[0]), int(tip[1])), 10, (0.65, 0.72, 1.0), -1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # Act V - VI: Pure Heat-Signature SDF Viper (Drops, Strikes, Coils -> Snatched!)
    # -------------------------------------------------------------------------

    def _viper_spine(self, t: float, cam_x: float) -> Optional[List[Tuple[float, float]]]:
        """Compute 20-point sinuous spine for the Act V-VI Pit Viper."""
        if t < 74.8 or t > 105.0:
            return None

        # Chameleon target on branch in screen coords: (1245 - 340, 388) = (905, 388)
        target_x, target_y = 1245.0 - cam_x, 388.0
        n = 20
        pts: List[Tuple[float, float]] = []

        # Phase 1: Drops sinuously from upper canopy (74.8s -> 81.2s)
        if t < 81.2:
            u = ease_out_cubic((t - 74.8) / 6.4)
            head_x = target_x + 110.0 * (1.0 - u) + 35.0 * math.sin(t * 2.5)
            head_y = lerp(-120.0, target_y - 95.0, u)
            for k in range(n):
                s = k / float(n - 1)
                sx = head_x + 55.0 * math.sin(s * 7.5 - t * 4.5) + s * 65.0
                sy = head_y - s * 410.0
                pts.append((sx, sy))

        # Phase 2: Violent predatory strike onto chameleon (81.2s -> 82.4s)
        elif t < 82.4:
            u = ease_in_cubic((t - 81.2) / 1.2)
            for k in range(n):
                s = k / float(n - 1)
                # Head rockets directly into chameleon while body whips down
                drop_x = target_x + 35.0 * math.sin(81.2 * 2.5) + 55.0 * math.sin(s * 7.5 - t * 4.5) + s * 65.0
                drop_y = (target_y - 95.0) - s * 410.0
                coil_ang = s * 11.5 - 1.5
                coil_x = target_x + 85.0 * math.cos(coil_ang) * (1.0 - 0.35 * s)
                coil_y = target_y + 52.0 * math.sin(coil_ang)
                pts.append((lerp(drop_x, coil_x, u), lerp(drop_y, coil_y, u)))

        # Phase 3: Distinct sinuous coils wrapping around the chameleon (Act V) -> menacing S-curve on branch (Act VI)
        elif t < 98.5:
            if t < 90.0:
                # Wide helical figure-8 wrap with raised head clearly distinct over chameleon!
                tighten = math.sin(t * 2.2) * 7.0
                for k in range(n):
                    s = k / float(n - 1)
                    if k < 4:
                        # Raised menacing pit-viper head peering over the wrapped prey
                        ang_h = -2.2 + k * 0.25
                        pts.append((target_x - 95.0 + k * 28.0, target_y - 78.0 + k * 14.0 + 6.0 * math.sin(t * 3.5)))
                    else:
                        sw = (k - 3) / float(n - 4)
                        coil_ang = sw * 10.2 + 0.4 + 0.25 * math.sin(t * 1.4)
                        rx = (132.0 - tighten) * (1.0 - 0.18 * sw)
                        ry = 68.0 - 0.35 * tighten
                        pts.append((target_x + rx * math.cos(coil_ang), target_y + 8.0 + ry * math.sin(coil_ang)))
            else:
                # Smoothly uncoil into a bold, unmistakable S-curved viper resting along the branch in Act VI!
                uncoil_u = smoothstep((t - 90.0) / 3.2)
                alarm = smoothstep((t - 95.0) / 2.2) if t > 95.0 else 0.0
                for k in range(n):
                    s = k / float(n - 1)
                    # Act V coiled end pose
                    coil_ang = s * 10.2 + 0.4
                    cx_v = target_x + 125.0 * (1.0 - 0.18 * s) * math.cos(coil_ang)
                    cy_v = target_y + 8.0 + 66.0 * math.sin(coil_ang)
                    # Act VI stretched S-curve along branch with head raised in alarm!
                    sx_v = target_x - 165.0 + s * 360.0
                    sy_v = target_y + 42.0 + 28.0 * math.sin(s * 7.2 - t * 2.5)
                    if k < 5:
                        sy_v -= (45.0 + 65.0 * alarm) * (1.0 - k / 5.0)
                    pts.append((lerp(cx_v, sx_v, uncoil_u), lerp(cy_v, sy_v, uncoil_u)))

        # Phase 4: Ripped skyward in the Golden Eagle's talons (98.5s -> 105.0s)!
        else:
            u = ease_in_cubic((t - 98.5) / 5.5)
            grab_x = lerp(target_x, target_x + 360.0, u)
            grab_y = lerp(target_y, -480.0, u)
            for k in range(n):
                s = k / float(n - 1)
                ds = s - 0.42
                sx = grab_x + ds * 195.0 + 48.0 * abs(ds) * math.sin(s * 11.0 - t * 18.0)
                sy = grab_y + abs(ds) * 320.0 + 22.0 * math.cos(s * 8.0 - t * 14.0)
                pts.append((sx, sy))
        return pts

    def _draw_sdf_viper(
        self,
        canvas: np.ndarray,
        heat_map: Optional[np.ndarray],
        t: float,
        light: LightingState,
        cam_x: float,
    ) -> None:
        """Render the SDF Viper (incandescent heat signature in Act V, Golden-Hour scales in Act VI)."""
        pts = self._viper_spine(t, cam_x)
        if pts is None:
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0, int(min(xs)) - 50)
        y0 = max(0, int(min(ys)) - 50)
        x1 = min(self.width, int(max(xs)) + 50)
        y1 = min(self.height, int(max(ys)) + 50)
        if x1 <= x0 + 8 or y1 <= y0 + 8:
            return

        X, Y = self.sdf.make_grid(x0, y0, x1, y1, scale=0.60)
        radii = [16.0 if i == 0 else (18.5 * math.sin(math.pi * clamp(i / 19.0 * 0.85 + 0.12)) ** 0.65) for i in range(len(pts))]
        radii[-1] = 4.5
        d = self.sdf.sd_polyline_tapered(X, Y, pts, radii, blend_k=5.5)

        # Angular pit-viper head seamlessly melted onto pts[0]
        hx, hy = pts[0]
        hang = math.atan2(pts[0][1] - pts[1][1], pts[0][0] - pts[1][0])
        d_head = self.sdf.sd_ellipse(X, Y, hx + 8 * math.cos(hang), hy + 8 * math.sin(hang), 26.0, 16.0, hang)
        d = self.sdf.smin(d, d_head, 6.5)

        nx, ny, nz, depth = self.sdf.compute_normals_3d(d, max_depth=22.0)
        # Optical emerald-olive diamond scales (visible in Act VI Golden Hour!)
        scales = 0.5 + 0.5 * np.sin((X + Y) * np.float32(0.18)) * np.cos((X - Y) * np.float32(0.18))
        albedo = np.stack(
            (0.08 + 0.08 * scales, 0.17 + 0.16 * scales, 0.12 + 0.16 * scales), axis=-1
        ).astype(np.float32)
        ndotl = np.clip(nx * light.light_dir[0] + ny * light.light_dir[1] + nz * light.light_dir[2], 0.0, 1.0)
        lit = albedo * (light.ambient_bgr * 0.6 + light.sun_color_bgr * (ndotl * light.sun_intensity * 1.1)[..., None])
        lit += self.sdf.shade_sss(d, nx, ny, nz, light.light_dir, sss_color_bgr=(0.10, 0.36, 1.00), sss_strength=0.72, sss_depth=12.0)

        # High-contrast Thermal Viper signature: blazing white-yellow spine core + orange-red edges so coils NEVER merge into a flat blob!
        head_glow = np.exp(-np.hypot(X - np.float32(hx), Y - np.float32(hy)) / np.float32(55.0))
        viper_heat = np.clip(np.float32(0.46) + np.power(depth, np.float32(0.75)) * np.float32(0.50) + head_glow * np.float32(0.12), 0.0, 1.0)

        self.sdf.composite_sdf(
            canvas, heat_map, (x0, y0, x1, y1), d, lit, heat_value=viper_heat, shadow_offset=(6, 10), shadow_alpha=0.28
        )

    # -------------------------------------------------------------------------
    # Act VI: Massive Golden Eagle with Directional High-Frequency Feather Noise
    # -------------------------------------------------------------------------

    def _draw_sdf_eagle(
        self,
        canvas: np.ndarray,
        t: float,
        light: LightingState,
        cam_x: float,
    ) -> None:
        """
        Render the massive Golden Eagle entering a sheer vertical dive in Act VI
        (93.0s -> 104.5s) using smooth SDF anatomy, directional high-frequency
        feather noise plumage, and intense Golden Hour backlit edge SSS!
        """
        if t < 93.0 or t > 105.0:
            return

        target_x, target_y = 1245.0 - cam_x, 360.0  # Impact right onto coiled viper!
        impact_t = 98.50

        if t <= impact_t:
            # Sheer vertical predatory stoop dive accelerating straight from the top!
            u = (t - 93.0) / (impact_t - 93.0)
            dive_u = 0.72 * ease_in_cubic(u) + 0.28 * u
            ex = target_x - 35.0 * (1.0 - dive_u)
            ey = lerp(-340.0, target_y - 30.0, dive_u)
            flare = smoothstep((u - 0.80) / 0.19)
            wing_tuck = 1.0 - flare
            flap = -0.30 * wing_tuck + 0.40 * flare
            pitch = 0.25 * (1.0 - flare)
        else:
            # Powerful roaring ascent carrying the snake off-top-right!
            u = clamp((t - impact_t) / 5.5)
            acc = ease_in_cubic(u)
            ex = lerp(target_x, target_x + 390.0, acc)
            ey = lerp(target_y - 30.0, -520.0, acc)
            wing_tuck = 0.0
            flap = math.sin((t - impact_t) * 16.0 - 0.7)
            pitch = -0.18 * smoothstep(u * 2.5)

        span = 470.0 - 155.0 * wing_tuck
        pad_x = int(span + 120)
        pad_y = 330
        x0, y0 = max(0, int(ex) - pad_x), max(0, int(ey) - pad_y)
        x1, y1 = min(self.width, int(ex) + pad_x), min(self.height, int(ey) + pad_y)
        if x1 <= x0 + 16 or y1 <= y0 + 16:
            return

        # 0.45x SDF grid gives crisp massive wings + high throughput
        X, Y = self.sdf.make_grid(x0, y0, x1, y1, scale=0.45)
        ca, sa = math.cos(pitch), math.sin(pitch)

        def to_w(lx: float, ly: float) -> Tuple[float, float]:
            return (ex + lx * ca - ly * sa, ey + lx * sa + ly * ca)

        # 1. Massive Aerodynamic Raptor Torso, Head & Hooked Beak
        d_torso = self.sdf.sd_ellipse(X, Y, ex, ey, 68.0, 92.0, pitch)
        hc = to_w(0.0, 72.0)
        d_head = self.sdf.sd_ellipse(X, Y, hc[0], hc[1], 36.0, 34.0, pitch)
        # Fierce curved hooked golden eagle beak
        bk0, bk1, bk2 = to_w(0.0, 96.0), to_w(14.0, 124.0), to_w(2.0, 138.0)
        d_beak = self.sdf.sd_polyline_tapered(X, Y, [bk0, bk1, bk2], [13.0, 9.0, 3.0], blend_k=4.0)
        d_eagle = self.sdf.smin(d_torso, d_head, 14.0)
        d_eagle = self.sdf.smin(d_eagle, d_beak, 6.0)

        # Fanned tail feathers behind torso
        t_c = to_w(0.0, -108.0)
        d_tail = self.sdf.sd_ellipse(X, Y, t_c[0], t_c[1], 88.0, 48.0, pitch)
        d_eagle = self.sdf.smin(d_eagle, d_tail, 16.0)

        # 2. Massive Swept Wings + Individual Sculpted Primary Flight Feathers
        tip_y = -185.0 - 95.0 * wing_tuck - 110.0 * flap
        mid_y = -65.0 - 40.0 * wing_tuck - 48.0 * flap
        for side in (-1.0, 1.0):
            w_pts = [
                to_w(side * 36.0, -10.0),
                to_w(side * span * 0.45, mid_y),
                to_w(side * span * 0.85, tip_y),
            ]
            d_wing = self.sdf.sd_polyline_tapered(X, Y, w_pts, [62.0, 56.0, 26.0], blend_k=22.0)
            # 5 distinct primary flight finger-feathers fanning off each wingtip!
            for pf in range(5):
                f_root = to_w(side * span * (0.78 - 0.05 * pf), tip_y + pf * 22.0)
                f_tip = to_w(side * span * (1.06 - 0.06 * pf), tip_y - 16.0 + pf * 32.0 - 20.0 * flap)
                d_pf = self.sdf.sd_capsule(X, Y, f_root[0], f_root[1], f_tip[0], f_tip[1], 11.0, 4.0)
                d_wing = self.sdf.smin(d_wing, d_pf, 6.0)
            d_eagle = self.sdf.smin(d_eagle, d_wing, 22.0)

        # Outstretched legs & talons thrusting downward at strike
        for side in (-1.0, 1.0):
            l0, l1 = to_w(side * 26.0, 42.0), to_w(side * 32.0, 118.0)
            d_leg = self.sdf.sd_capsule(X, Y, l0[0], l0[1], l1[0], l1[1], 14.0, 8.0)
            d_eagle = self.sdf.smin(d_eagle, d_leg, 8.0)

        # 3. Directional High-Frequency Feather Noise Plumage + Golden Hour SSS!
        nx, ny, nz, depth = self.sdf.compute_normals_3d(d_eagle, max_depth=46.0)
        # Symmetric left/right directional high-frequency barb texture
        barb_ang = np.where(X < np.float32(ex), np.float32(0.45 + pitch), np.float32(-0.45 + pitch))
        fn_l = self.sdf.directional_feather_noise(X, Y, ex, ey, 0.45 + pitch)
        fn_r = self.sdf.directional_feather_noise(X, Y, ex, ey, -0.45 + pitch)
        f_noise = np.where(X < np.float32(ex), fn_l, fn_r)

        # Deep espresso-chocolate wing feathers transitioning into radiant golden head hackles!
        head_dist = np.hypot(X - np.float32(hc[0]), Y - np.float32(hc[1]))
        golden_nape = np.exp(-head_dist / np.float32(82.0))
        wing_dark = np.clip(np.abs(X - np.float32(ex)) / np.float32(span * 0.95), 0.0, 1.0)
        plumage_b = np.float32(0.04) + np.float32(0.09) * f_noise * (1.0 - 0.45 * wing_dark) + np.float32(0.05) * golden_nape
        plumage_g = np.float32(0.08) + np.float32(0.20) * f_noise * (1.0 - 0.45 * wing_dark) + np.float32(0.26) * golden_nape
        plumage_r = np.float32(0.14) + np.float32(0.32) * f_noise * (1.0 - 0.38 * wing_dark) + np.float32(0.42) * golden_nape
        albedo = np.stack((plumage_b, plumage_g, plumage_r), axis=-1)

        # Bright golden-amber eagle beak tint
        beak_mask = smoothstep_arr(3.0, -3.0, d_beak)[..., None]
        albedo = albedo * (1.0 - beak_mask) + np.array([0.14, 0.72, 0.98], dtype=np.float32) * beak_mask

        ndotl = np.clip(nx * light.light_dir[0] + ny * light.light_dir[1] + nz * light.light_dir[2], 0.0, 1.0)
        lit = albedo * (light.ambient_bgr * 0.65 + light.sun_color_bgr * (ndotl * light.sun_intensity * 1.15)[..., None])
        # Fiery Golden Hour backlit red/orange SSS glow on every wing feather edge!
        lit += self.sdf.shade_sss(
            d_eagle, nx, ny, nz, light.light_dir,
            sss_color_bgr=(0.12, 0.42, 1.00),
            sss_strength=1.05,
            sss_depth=22.0,
        )

        self.sdf.composite_sdf(
            canvas, None, (x0, y0, x1, y1), d_eagle, lit, shadow_offset=(18, 28), shadow_alpha=0.28
        )

        # Piercing golden-amber raptor eyes on the head
        for side in (-1.0, 1.0):
            ec = to_w(side * 14.0, 74.0)
            cv2.circle(canvas, (int(ec[0]), int(ec[1])), 6, (0.12, 0.78, 1.0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(ec[0]), int(ec[1])), 3, (0.02, 0.02, 0.03), -1, cv2.LINE_AA)

        # Subtle impact flash burst at exact impact moment (98.50s -> 98.85s)
        if 98.50 <= t <= 98.90:
            fl = math.exp(-((t - 98.52) / 0.14) ** 2)
            canvas += light.sun_color_bgr * np.float32(fl * 0.48)

    # -------------------------------------------------------------------------
    # Act VII: Single Hero Feather Landing & Bioluminescent Rebirth (110s-120s)
    # -------------------------------------------------------------------------

    def _draw_act7_rebirth(
        self, canvas: np.ndarray, t: float, light: LightingState, cam_x: float
    ) -> None:
        """
        Single golden eagle feather drifts gently onto the dark night ground,
        then cyan bioluminescent L-system fungi from Act I blooms directly out
        of its quill — completing The Circle of Life!
        """
        if t < 109.5:
            return

        # Hero feather floats down (109.5s -> 113.2s) and rests on dark soil at (640, 595)
        land_u = clamp((t - 109.5) / 3.7)
        e_u = ease_out_cubic(land_u)
        fx = 640.0 - cam_x + 75.0 * (1.0 - e_u) * math.sin(t * 1.9)
        fy = lerp(85.0, 598.0, e_u)
        ang = -0.18 + 0.45 * (1.0 - e_u) * math.cos(t * 1.9)

        # Cyan bioluminescent energy infusing the feather quill starting at 113.0s!
        grow = clamp((t - 113.0) / 5.2)
        glow_q = smoothstep((t - 112.8) / 1.8)

        self.particles.draw_procedural_feather(
            canvas,
            (fx, fy),
            length=155.0,
            angle=ang,
            shade=lerp(0.88, 0.55, smoothstep((t - 110.0) / 4.0)),
            alpha=0.96,
            rim_bgr=lerp_vec(light.sun_color_bgr, (1.0, 0.92, 0.25), glow_q),
            glow_quill=glow_q,
        )

        # Bioluminescent cyan L-system mycelium erupts directly FROM THE QUILL & wraps over it!
        if grow > 0.01:
            quill_x = fx - 55.0 * math.cos(ang) + cam_x
            quill_y = fy - 55.0 * math.sin(ang)
            mid_x = fx + cam_x
            mid_y = fy
            rebirth_origins = [
                (quill_x, quill_y, 18.0, -75.0),
                (quill_x + 18.0, quill_y + 4.0, 16.0, -25.0),
                (mid_x, mid_y, 17.0, -110.0),
                (mid_x + 28.0, mid_y - 4.0, 14.0, -45.0),
            ]
            self._draw_mycelium(
                canvas, t, cam_x, rebirth_origins, growth=grow, glow_strength=light.bio_glow_strength
            )

    # -------------------------------------------------------------------------
    # Master Frame Pipeline (Single Frame 0..2879)
    # -------------------------------------------------------------------------

    def render_frame_rgb(self, frame_idx: int) -> np.ndarray:
        """Render one complete 1280x720 RGB uint8 frame of 'The Circle of Life'."""
        t = frame_idx / float(self.fps)
        light = self.lighting.evaluate(t)
        cam_x, cam_y = self._camera_state(t)

        # 1. Volumetric Sky, Global Illumination, God Rays & Jungle Environment
        canvas, heat_map = self.lighting.render_environment(t, light, cam_x, cam_y)

        # 2. Act I Bioluminescent Cyan Mycelium (0s -> 22s, dimming as sunrise arrives)
        if t < 23.0:
            myc_grow = clamp((t - 0.5) / 11.5)
            act1_origins = [
                (235.0, 635.0, 21.0, -82.0),
                (445.0, 618.0, 23.0, -96.0),
                (680.0, 642.0, 22.0, -78.0),
                (915.0, 628.0, 20.0, -104.0),
                (1080.0, 648.0, 19.0, -86.0),
            ]
            self._draw_mycelium(
                canvas, t, cam_x, act1_origins, growth=myc_grow, glow_strength=light.bio_glow_strength
            )

        # 3. Act II-IV Fractal Alien Fern (Fibonacci spiral unrolling)
        eaten = clamp((t - 42.0) / 8.0) if t > 42.0 else 0.0
        self._draw_alien_fern(canvas, t, light, cam_x, eaten_amount=eaten)

        # 4. Act IV-VI Mossy Tree Branch
        self._draw_tree_branch(canvas, heat_map, t, light, cam_x)

        # 5. Act III-IV SDF Iridescent Rhinoceros Beetle (6-leg analytical IK)
        self._draw_sdf_beetle(canvas, heat_map, t, light, cam_x)

        # 6. Act IV-V SDF Voronoi Camouflage Chameleon + Independent Eyes + Tongue
        self._draw_sdf_chameleon(canvas, heat_map, t, light, cam_x)

        # 7. Act V-VI SDF Pit Viper (Heat signature -> Coils -> Snatched!)
        self._draw_sdf_viper(canvas, heat_map, t, light, cam_x)

        # 8. Act VI Massive SDF Golden Eagle (Directional feather noise + Golden Hour SSS)
        self._draw_sdf_eagle(canvas, t, light, cam_x)

        # 9. Act VII Dust to Dust: Single Feather + Bioluminescent Fungal Rebirth
        self._draw_act7_rebirth(canvas, t, light, cam_x)

        # 10. Vectorized Particle Systems (Rain, Ripples, Spores, Pollen, 3D Lens Feathers)
        if light.rain_intensity > 0.01:
            self.particles.render_rain_and_ripples(canvas, t, light.rain_intensity, light.bio_glow_strength)
        if light.bio_glow_strength > 0.05:
            self.particles.render_bio_spores(canvas, t, light.bio_glow_strength, cam_x)
        if 17.5 <= t <= 69.5:
            pol_s = smoothstep((t - 17.5) / 3.0) * (1.0 - smoothstep((t - 66.5) / 3.0))
            self.particles.render_pollen(canvas, t, pol_s, cam_x, light.sun_color_bgr)
        if 98.50 <= t <= 110.5:
            self.particles.render_feather_explosion(
                canvas, t, 98.50, (1245.0 - cam_x, 380.0), light.sun_color_bgr
            )

        # 11. Act V Full-Frame Thermal Heatmap Vision Post-Processing Filter
        if light.thermal_mix > 0.005 and heat_map is not None:
            canvas = self.lighting.apply_thermal_filter(canvas, heat_map, light.thermal_mix, t)

        # 12. Subtle Filmic ACES-like Tone Curve + Cinematic Fade In/Out
        canvas = np.clip(canvas, 0.0, 1.8)
        canvas = (canvas * (np.float32(2.51) * canvas + np.float32(0.03))) / (
            canvas * (np.float32(2.43) * canvas + np.float32(0.59)) + np.float32(0.14)
        )
        canvas = np.clip(canvas * np.float32(light.fade_black), 0.0, 1.0)

        bgr_u8 = (canvas * np.float32(255.0)).astype(np.uint8)
        return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------------------
    # Parallel Multiprocessing Chunk Renderer & MoviePy MP4 Assembly
    # -------------------------------------------------------------------------

    def render_movie(
        self,
        output_path: str = "the_circle_of_life_120s.mp4",
        workers: Optional[int] = None,
        chunk_size: int = CHUNK_SIZE,
    ) -> str:
        """
        Render all 2,880 frames in parallel chunks via `multiprocessing` and
        assemble with the generative binaural DSP soundtrack into an MP4.
        """
        n_workers = workers or max(1, os.cpu_count() or 2)
        tmp_wav_fd, tmp_wav_path = tempfile.mkstemp(suffix="_circle_of_life.wav")
        os.close(tmp_wav_fd)

        audio_clip = None
        video_clip = None
        try:
            print(f"[1/3] Synthesizing {DURATION:.1f}s Brian Eno ambient score + binaural foley...")
            audio_synth = DSP_Audio_Synth(self.particles, duration=DURATION, sample_rate=SAMPLE_RATE)
            audio_synth.export_wav(tmp_wav_path)

            print(
                f"[2/3] Rendering {TOTAL_FRAMES} procedural SDF frames "
                f"({WIDTH}x{HEIGHT} @ {FPS} FPS) across {n_workers} worker processes..."
            )
            chunks = [
                (start, min(TOTAL_FRAMES, start + chunk_size))
                for start in range(0, TOTAL_FRAMES, chunk_size)
            ]

            with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
                # Bounded prefetch window keeps RAM compact while saturating all CPU cores!
                max_inflight = n_workers * 2
                pending = []
                next_submit = 0
                current_chunk_start = -1
                current_chunk_frames: List[np.ndarray] = []

                def refill():
                    nonlocal next_submit
                    while len(pending) < max_inflight and next_submit < len(chunks):
                        pending.append((chunks[next_submit], pool.submit(_render_chunk, chunks[next_submit])))
                        next_submit += 1

                refill()

                def make_frame(t_sec: float) -> np.ndarray:
                    nonlocal current_chunk_start, current_chunk_frames
                    fidx = min(max(0, int(round(t_sec * FPS))), TOTAL_FRAMES - 1)
                    while not (current_chunk_start <= fidx < current_chunk_start + len(current_chunk_frames)):
                        if not pending:
                            # Fallback if MoviePy requests an out-of-order frame (e.g., frame 0 probe)
                            return self.render_frame_rgb(fidx)
                        (c_start, _), fut = pending.pop(0)
                        current_chunk_start = c_start
                        current_chunk_frames = fut.result()
                        refill()
                    return current_chunk_frames[fidx - current_chunk_start]

                # MoviePy probes t=0 during VideoClip init, so keep frame 0 clean without consuming queue
                frame0 = self.render_frame_rgb(0)

                def safe_make_frame(t_sec: float) -> np.ndarray:
                    fidx = min(max(0, int(round(t_sec * FPS))), TOTAL_FRAMES - 1)
                    if fidx == 0 and current_chunk_start == -1:
                        return frame0
                    return make_frame(t_sec)

                video_clip = VideoClip(safe_make_frame, duration=DURATION)
                audio_clip = AudioFileClip(tmp_wav_path)
                final_clip = (
                    video_clip.set_audio(audio_clip)
                    if hasattr(video_clip, "set_audio")
                    else video_clip.with_audio(audio_clip)
                )

                print(f"[3/3] Encoding cinematic MP4 to {output_path} ...")
                final_clip.write_videofile(
                    output_path,
                    fps=FPS,
                    codec="libx264",
                    audio_codec="aac",
                    audio_bitrate="192k",
                    preset="medium",
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18"],
                    threads=n_workers,
                )

            print(f"Done! Rendered {output_path} ({DURATION:.1f}s, {TOTAL_FRAMES} frames).")
            return output_path
        finally:
            if audio_clip is not None:
                audio_clip.close()
            if video_clip is not None:
                video_clip.close()
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)


# Global worker singleton initialized once per multiprocessing child process
_WORKER_CINEMA: Optional[Cinematographer] = None


def _init_worker() -> None:
    global _WORKER_CINEMA
    _WORKER_CINEMA = Cinematographer(WIDTH, HEIGHT, FPS)


def _render_chunk(bounds: Tuple[int, int]) -> List[np.ndarray]:
    """Top-level picklable multiprocessing worker rendering a chunk of frames."""
    assert _WORKER_CINEMA is not None
    start_f, end_f = bounds
    return [_WORKER_CINEMA.render_frame_rgb(f) for f in range(start_f, end_f)]


if __name__ == "__main__":
    cinema = Cinematographer(width=WIDTH, height=HEIGHT, fps=FPS)
    cinema.render_movie(output_path="the_circle_of_life_120s.mp4")
