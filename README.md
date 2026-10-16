# Automatizált adatelőfeldolgozó és elemző pipeline

Egyetemi Python projekt laboratóriumi mérésekhez: **mikroszkópos képszegmentálás**, **spektroszkópiai / idősoros jelfeldolgozás** és **interaktív vizualizáció**. A manuális kiértékelés lassú és hibaveszélyes; a pipeline ezeket a lépéseket reprodukálhatóvá és paraméterezhetővé teszi.

## Cél

A rendszer három, a gyakorlatban gyakran együtt előforduló adatforrást kezel:

| Modul | Bemenet | Feladat |
| --- | --- | --- |
| Képszegmentálás | fluoreszcens / fénymikroszkópos felvétel | egyedi sejtek, szöveti mintázatok azonosítása deep learninggel (U-Net) |
| Jelfeldolgozás | idősor, FTIR-szerű spektrum, NMR-szerű FID | zajszűrés, fázisjavítás, spektrális elemzés |
| Vizualizáció | 2D intenzitásmező vagy szegmentált maszk | hőtérkép, 3D felület, állítható küszöbök |

A demó **szintetikus, de fizikailag motivált** adatokat is előállít, így a kód valós minták nélkül is futtatható és bemutatható.

## Architektúra

```
python_projekt/
├── app.py                      # Streamlit felhasználói felület
├── scripts/train_unet.py       # U-Net tanítás
├── src/bioanalysis/
│   ├── segmentation/           # U-Net, szintetikus sejtképek, inferencia
│   ├── signals/                # szűrők, fázis, FFT / spektrogram
│   └── visualization/          # hőtérkép és küszöbmaszk
├── data/models/                # betanított súlyok
└── tests/                      # automatikus ellenőrzések
```

A mélytanulásos szegmentáló egy **kompakt U-Net** (kódoló–dekódoló skip connectionökkel, BCE + Dice veszteség). CPU-n, néhány epoch alatt tanítható szintetikus fluoreszcens képeken. Ha a súlyfájl hiányzik, a pipeline **Otsu-küszöbölésre** vált, így a bemutató akkor is működik.

## Követelmények

- Python 3.11+
- ajánlott virtuális környezet

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS aktiválás: `source .venv/bin/activate`

## Futtatás

Interaktív felület:

```bash
python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run app.py
```

U-Net tanítása (opcionális, de a deep learninges szegmentáláshoz kell):

```bash
python scripts/train_unet.py
```

A súlyok helye: `data/models/unet_cells.pt`.

Tesztek:

```bash
pytest -q
```

## Módszerek röviden

### Képszegmentálás

- Szintetikus sejtgenerátor: Gauss-blobok, egyenetlen megvilágítás, additív zaj.
- U-Net: 16–128 csatorna, 3 felbontási szint, szigmoid kimenet.
- Objektumszámlálás: összefüggő komponensek (`skimage.measure.label`) és `regionprops`.
- Paraméterek a felületen: valószínűségi küszöb, minimális objektumméret.

### Jelfeldolgozás

- **Zajszűrés:** Savitzky–Golay, Butterworth aluláteresztő, Wiener.
- **Fázisjavítás:** 0. és 1. rendű fázis NMR-szerű FID-en; becslés a valós spektrum entrópiájának és negativitásának minimalizálásával.
- **Spektrális elemzés:** FFT / rFFT, spektrogram-hőtérkép, csúcskeresés prominencia alapján.

### Vizualizáció

- Plotly hőtérkép a küszöbtartományon belüli pixelekkel.
- 3D `Surface` az intenzitásmezőre.
- A küszöb csúszkái élőben szűrik a megjelenített tartományt.

## Saját adatok

- Mikroszkópos kép: a **Képszegmentálás** lapon tölthető fel PNG/JPG/TIFF.
- Saját számidősor vagy spektrum a `bioanalysis.signals` függvényeivel dolgozható fel Pythonból, például:

```python
from bioanalysis.signals.filters import denoise_signal
from bioanalysis.signals.spectral import power_spectrum, peak_table

clean = denoise_signal(y, method="savgol", window=21)
freq, spec = power_spectrum(clean, fs=100.0)
peaks = peak_table(freq, abs(spec))
```

## Korlátok (őszintén)

Ez oktatási demó, nem klinikai eszköz. A U-Net szintetikus blobokon tanul, ezért valódi patológiai metszeteken finomhangolás és annotált adat kell. A fázisbecslés egyszerű heurisztika, nem teljes NMR-feldolgozó lánc.

## Licenc

Oktatási felhasználásra készült egyetemi beadandó.
