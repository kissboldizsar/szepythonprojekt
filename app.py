"""Interaktív elemző felület a laboratóriumi pipeline-hoz."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bioanalysis.segmentation.infer import count_objects, segment_image
from bioanalysis.segmentation.synthetic import generate_cell_image
from bioanalysis.signals.filters import denoise_signal
from bioanalysis.signals.phase import apply_phase_correction, estimate_phase
from bioanalysis.signals.spectral import peak_table, power_spectrum, spectrogram
from bioanalysis.signals.synthetic import generate_fid, generate_spectrum, generate_timeseries
from bioanalysis.visualization.maps import intensity_grid, overlay_mask, threshold_mask

st.set_page_config(page_title="Bioanalízis pipeline", layout="wide")
st.title("Automatizált adatelőfeldolgozó és elemző pipeline")
st.caption("Mikroszkópos szegmentálás • spektroszkópia / idősor • interaktív hőtérkép és 3D felület")

page = st.sidebar.radio(
    "Modul",
    ["Képszegmentálás", "Jelfeldolgozás", "Hőtérkép és 3D"],
)


def _upload_to_array(uploaded) -> np.ndarray:
    image = Image.open(uploaded).convert("L")
    return np.asarray(image, dtype=np.float32)


if page == "Képszegmentálás":
    st.header("Mikroszkópos képszegmentálás")
    st.write(
        "A modell egy kompakt U-Net, szintetikus fluoreszcens sejtképeken tanítva. "
        "Ha a súlyfájl még nincs betanítva, a rendszer Otsu-küszöbölésre vált."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded = st.file_uploader("Saját kép (PNG/JPG)", type=["png", "jpg", "jpeg", "tif", "tiff"])
        n_cells = st.slider("Szintetikus sejtek száma", 5, 30, 14)
        seed = st.number_input("Véletlen mag", 0, 9999, 7)
    with col_b:
        threshold = st.slider("Valószínűségi küszöb", 0.05, 0.95, 0.45, 0.01)
        min_size = st.slider("Minimális objektumméret (px)", 5, 120, 20)
        prefer_unet = st.checkbox("U-Net használata, ha elérhető", value=True)

    if uploaded is not None:
        image = _upload_to_array(uploaded)
        true_n = None
    else:
        sample = generate_cell_image(size=128, n_cells=int(n_cells), seed=int(seed))
        image = sample.image
        true_n = sample.n_cells

    result = segment_image(image, threshold=float(threshold), min_size=int(min_size), prefer_unet=prefer_unet)
    overlay = overlay_mask(image, result.mask)
    objects = count_objects(result)

    m1, m2, m3 = st.columns(3)
    m1.metric("Detektált minták", result.n_objects)
    m2.metric("Módszer", result.method.upper())
    m3.metric("Valós sejtszám (szintetikus)", true_n if true_n is not None else "—")

    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="Bemeneti felvétel", use_container_width=True, clamp=True)
    c2.image(result.probability, caption="Valószínűség / intenzitás", use_container_width=True, clamp=True)
    c3.image(overlay, caption="Szegmentált maszk", use_container_width=True, clamp=True)

    if objects:
        st.dataframe(objects, use_container_width=True)

elif page == "Jelfeldolgozás":
    st.header("Zajszűrés, fázisjavítás és spektrális elemzés")
    kind = st.selectbox("Adattípus", ["idősor", "spektrum", "FID (NMR-szerű)"])
    noise = st.slider("Zajszint", 0.0, 0.8, 0.25, 0.01)
    method = st.selectbox("Szűrő", ["savgol", "butterworth", "wiener"])
    window = st.slider("Szűrőablak", 5, 51, 21, 2)

    if kind == "idősor":
        rec = generate_timeseries(noise=float(noise))
        y_f = denoise_signal(rec.y, method=method, window=window)
        freq, spec = power_spectrum(y_f, fs=float(rec.extra["fs"]))
        f, t, sxx = spectrogram(y_f, fs=float(rec.extra["fs"]))
        peaks = peak_table(freq, np.abs(spec))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec.x, y=rec.y, name="nyers", opacity=0.45))
        fig.add_trace(go.Scatter(x=rec.x, y=y_f, name="szűrt"))
        fig.update_layout(title="Idősor", xaxis_title="t [s]", yaxis_title="amplitúdó", height=360)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Scatter(x=freq, y=np.abs(spec), name="|FFT|"))
        fig2.update_layout(title="Teljesítményspektrum", xaxis_title="f [Hz]", height=320)
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure(go.Heatmap(z=10 * np.log10(sxx + 1e-12), x=t, y=f, colorscale="Viridis"))
        fig3.update_layout(title="Spektrogram (hőtérkép)", xaxis_title="t [s]", yaxis_title="f [Hz]", height=360)
        st.plotly_chart(fig3, use_container_width=True)
        st.subheader("Csúcsok")
        st.dataframe(peaks, use_container_width=True)

    elif kind == "spektrum":
        rec = generate_spectrum(noise=float(noise))
        y_f = denoise_signal(rec.y, method=method, window=window)
        peaks = peak_table(rec.x, y_f)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec.x, y=rec.y, name="nyers", opacity=0.4))
        fig.add_trace(go.Scatter(x=rec.x, y=y_f, name="szűrt"))
        fig.update_layout(title="FTIR-szerű spektrum", xaxis_title="hullámszám [cm⁻¹]", yaxis_title="abszorbancia", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(peaks, use_container_width=True)

    else:
        rec = generate_fid(noise=float(noise))
        y_d = denoise_signal(rec.y, method=method, window=window)
        phi0, phi1 = estimate_phase(y_d)
        corrected = apply_phase_correction(y_d, phi0, phi1)
        fs = float(rec.extra["fs"])
        freq_raw, spec_raw = power_spectrum(rec.y, fs=fs)
        freq, spec = power_spectrum(corrected, fs=fs)

        c1, c2 = st.columns(2)
        c1.metric("φ₀ [rad]", f"{phi0:.3f}")
        c2.metric("φ₁ [rad]", f"{phi1:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec.x, y=rec.y.real, name="FID Re", opacity=0.5))
        fig.add_trace(go.Scatter(x=rec.x, y=corrected.real, name="javított Re"))
        fig.update_layout(title="Szabad indukciós lecsengés", xaxis_title="t [s]", height=360)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=freq_raw, y=spec_raw.real, name="fázishibás Re(FFT)", opacity=0.45))
        fig2.add_trace(go.Scatter(x=freq, y=spec.real, name="fázisjavított Re(FFT)"))
        fig2.update_layout(title="Spektrum fázisjavítás után", xaxis_title="f [Hz]", height=360)
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.header("Interaktív hőtérkép és 3D felület")
    size = st.slider("Rácsméret", 32, 128, 64, 8)
    seed = st.number_input("Mező mag", 0, 9999, 3)
    low, high = st.slider("Intenzitás-küszöb tartomány", 0.0, 1.5, (0.25, 1.2))
    grid = intensity_grid(size=int(size), seed=int(seed))
    grid = grid / (grid.max() + 1e-8)
    mask = threshold_mask(grid, float(low), float(high))
    masked = np.where(mask, grid, np.nan)

    st.metric("Küszöbön belüli pixelek aránya", f"{100 * mask.mean():.1f} %")

    fig_h = go.Figure(go.Heatmap(z=masked, colorscale="Hot", colorbar=dict(title="intenzitás")))
    fig_h.update_layout(title="Hőtérkép küszöböléssel", height=420)
    st.plotly_chart(fig_h, use_container_width=True)

    fig_s = go.Figure(go.Surface(z=grid, colorscale="Viridis", showscale=True))
    fig_s.update_layout(
        title="3D intenzitásfelület",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="I", aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.45)),
        height=520,
    )
    st.plotly_chart(fig_s, use_container_width=True)
