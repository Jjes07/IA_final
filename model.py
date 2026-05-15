import librosa
import numpy as np
import torch
from transformers import pipeline
import matplotlib
matplotlib.use('Agg')  # backend sin pantalla para Gradio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from PIL import Image

# ── Categorías de instrumentos ────────────────────────────
INSTRUMENT_KEYWORDS = [
    "guitar", "piano", "violin", "drum", "bass", "trumpet", "saxophone",
    "flute", "cello", "synthesizer", "keyboard", "banjo", "mandolin",
    "ukulele", "harp", "trombone", "clarinet", "organ", "percussion",
    "harmonica", "accordion", "electric guitar", "acoustic guitar",
    "bass guitar", "steel guitar", "double bass"
]

INSTRUMENT_LABELS_ES = {
    "cel": "Cello",
    "cello": "Cello",
    "gac": "Guitarra acústica",
    "acoustic guitar": "Guitarra acústica",
    "gel": "Guitarra eléctrica",
    "electric guitar": "Guitarra eléctrica",
    "guitar": "Guitarra",
    "pia": "Piano",
    "piano": "Piano",
    "sax": "Saxofón",
    "saxophone": "Saxofón",
    "tru": "Trompeta",
    "trumpet": "Trompeta",
    "vio": "Violín",
    "violin": "Violín",
    "voi": "Voz",
    "voice": "Voz",
    "singing": "Voz",
    "organ": "Órgano",
    "bass": "Bajo",
    "bass guitar": "Bajo eléctrico",
    "drums": "Batería",
    "drum": "Batería",
    "flute": "Flauta",
    "harpsichord": "Clavecín",
    "piano, electric piano": "Piano eléctrico",
}

# ── Carga del modelo ──────────────────────────────────────────────────────────
print("Cargando modelo...")
classifier = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
    device=0 if torch.cuda.is_available() else -1
)
print(f"Modelo cargado en: {'GPU' if torch.cuda.is_available() else 'CPU'}")


def is_instrument(label: str) -> bool:
    label_lower = label.lower()
    return any(kw in label_lower for kw in INSTRUMENT_KEYWORDS)


def analyze_segment(audio_segment: np.ndarray, sr: int) -> list[dict]:
    results = classifier(
        {"raw": audio_segment, "sampling_rate": sr},
        top_k=527
    )
    
    instruments = [
        {"label": r["label"], "score": r["score"]}
        for r in results
        if is_instrument(r["label"])
    ]
    
    if not instruments:
        return []
    
    # Renormalizar solo entre instrumentos
    total = sum(i["score"] for i in instruments)
    for inst in instruments:
        inst["score"] = inst["score"] / total
    
    # Umbral del 3% para no descartar demasiado
    instruments = [i for i in instruments if i["score"] >= 0.03]
    
    return sorted(instruments, key=lambda x: x["score"], reverse=True)


def analyze_audio(audio_path: str, segment_duration: int = 5) -> dict:
    """
    Pipeline completo de análisis:
    1. Carga el audio
    2. Lo segmenta en ventanas de segment_duration segundos
    3. Analiza cada segmento
    4. Agrega resultados globales
    """
    # Cargar audio
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_duration = librosa.get_duration(y=y, sr=sr)

    segment_samples = segment_duration * sr
    n_segments = max(1, int(total_duration // segment_duration))

    # Limitar a los primeros 60 segundos para no tardar demasiado
    n_segments = min(n_segments, 60 // segment_duration)

    timeline = []      # lista de (tiempo_inicio, tiempo_fin, instrumentos)
    all_scores = {}    # acumulador para el resultado global

    for i in range(n_segments):
        start = i * segment_samples
        end   = min(start + segment_samples, len(y))
        segment = y[start:end]

        if len(segment) < sr:   # segmento demasiado corto, ignorar
            continue

        instruments = analyze_segment(segment, sr)

        t_start = i * segment_duration
        t_end   = min(t_start + segment_duration, total_duration)
        timeline.append({
            "t_start":     t_start,
            "t_end":       t_end,
            "instruments": instruments
        })

        for inst in instruments:
            label = inst["label"]
            if label not in all_scores:
                all_scores[label] = []
            all_scores[label].append(inst["score"])

    # Promedio de confianza por instrumento en toda la canción
    global_results = {
        label: float(np.mean(scores))
        for label, scores in all_scores.items()
    }
    # Ordenar de mayor a menor confianza
    global_results = dict(
        sorted(global_results.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "duration":       total_duration,
        "n_segments":     n_segments,
        "global_results": global_results,
        "timeline":       timeline,
    }


def build_bar_chart(global_results: dict) -> Image.Image:
    """Genera gráfico de barras de confianza por instrumento."""
    if not global_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No se detectaron instrumentos",
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        top = dict(list(global_results.items())[:10])
        labels = [INSTRUMENT_LABELS_ES.get(k.lower(), k) for k in top.keys()]
        scores = [v * 100 for v in top.values()]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
        bars = ax.barh(labels, scores, color=colors)
        ax.set_xlabel("Confianza relativa entre instrumentos detectados (%)")
        ax.set_title("Instrumentos detectados")
        ax.set_xlim(0, 100)
        ax.bar_label(bars, fmt="%.1f%%", padding=4)
        ax.invert_yaxis()
        fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def build_timeline_chart(timeline: list, duration: float) -> Image.Image:
    """Genera gráfico de presencia temporal de instrumentos."""
    # Recopilar todos los instrumentos únicos
    all_instruments = set()
    for seg in timeline:
        for inst in seg["instruments"]:
            if inst["score"] > 0.05:
                all_instruments.add(inst["label"])

    if not all_instruments:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "Sin datos de timeline",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    instruments_list = sorted(all_instruments)
    cmap = plt.cm.tab20
    colors = {inst: cmap(i / len(instruments_list))
              for i, inst in enumerate(instruments_list)}

    fig, ax = plt.subplots(figsize=(12, max(3, len(instruments_list) * 0.5 + 2)))

    for y_pos, instrument in enumerate(instruments_list):
        for seg in timeline:
            for inst in seg["instruments"]:
                if inst["label"] == instrument and inst["score"] > 0.05:
                    alpha = min(1.0, inst["score"] * 3)
                    ax.barh(
                        y_pos,
                        seg["t_end"] - seg["t_start"],
                        left=seg["t_start"],
                        color=colors[instrument],
                        alpha=alpha,
                        edgecolor='white',
                        linewidth=0.5
                    )

    label_names = [INSTRUMENT_LABELS_ES.get(i.lower(), i) for i in instruments_list]
    ax.set_yticks(range(len(instruments_list)))
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_title("Presencia temporal de instrumentos")
    ax.set_xlim(0, min(duration, len(timeline) * 5))
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def format_results_for_bot(results: dict) -> str:
    """Serializa los resultados para pasarlos al bot como contexto."""
    if not results:
        return "No hay resultados de análisis aún."

    lines = [
        f"Duración del audio: {results['duration']:.1f} segundos",
        f"Segmentos analizados: {results['n_segments']}",
        "",
        "Instrumentos detectados (confianza promedio):"
    ]
    for label, score in results["global_results"].items():
        nombre = INSTRUMENT_LABELS_ES.get(label.lower(), label)
        lines.append(f"  - {nombre}: {score*100:.1f}%")

    return "\n".join(lines)