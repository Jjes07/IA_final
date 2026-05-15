import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import time
from imblearn.over_sampling import SMOTE

# ── Configuración ─────────────────────────────────────────────────────────────
IRMAS_PATH    = "dataset/IRMAS-TrainingData"
MODEL_PATH    = "irmas_model.pkl"
ENCODER_PATH  = "irmas_encoder.pkl"
FEATURES_CACHE = "dataset/irmas_features.npz"

INSTRUMENT_NAMES = {
    "cel": "Cello",
    "cla": "Clarinete",
    "flu": "Flauta",
    "gac": "Guitarra acústica",
    "gel": "Guitarra eléctrica",
    "org": "Órgano",
    "pia": "Piano",
    "sax": "Saxofón",
    "tru": "Trompeta",
    "vio": "Violín",
    "voi": "Voz"
}


# ── Extracción de features ────────────────────────────────────────────────────

def extract_features(audio_path: str, sr: int = 22050) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=3.0)

    mfccs       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean  = np.mean(mfccs, axis=1)
    mfccs_std   = np.std(mfccs, axis=1)

    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mean  = np.mean(delta_mfccs, axis=1)

    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    contrast      = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    return np.concatenate([
        mfccs_mean, mfccs_std, delta_mean,
        chroma_mean, contrast_mean,
        [zcr, rms]
    ])


def extract_features_safe(fpath: str, instrument: str):
    """Wrapper para paralelización — devuelve None si falla."""
    try:
        return extract_features(fpath), instrument
    except Exception:
        return None, None


# ── Carga del dataset con caché ───────────────────────────────────────────────

def load_irmas_dataset() -> tuple[np.ndarray, np.ndarray]:
    # Si existe el caché, cargarlo directamente (segundos en lugar de minutos)
    if os.path.exists(FEATURES_CACHE):
        print("✅ Cargando features desde caché...")
        data = np.load(FEATURES_CACHE, allow_pickle=True)
        X, y = data['X'], data['y']
        print(f"   {len(X)} muestras cargadas desde caché.")
        return X, y

    # Primera vez: extraer features en paralelo
    tasks = []
    instruments = sorted(os.listdir(IRMAS_PATH))
    print(f"Instrumentos encontrados: {instruments}\n")

    for instrument in instruments:
        instrument_path = os.path.join(IRMAS_PATH, instrument)
        if not os.path.isdir(instrument_path):
            continue
        files = [f for f in os.listdir(instrument_path)
                 if f.endswith(('.wav', '.mp3', '.aiff'))]
        print(f"  {instrument}: {len(files)} archivos")
        for fname in files:
            tasks.append((os.path.join(instrument_path, fname), instrument))

    print(f"\nExtrayendo features de {len(tasks)} archivos en paralelo (n_jobs=-1)...")
    t0 = time.time()

    results = Parallel(n_jobs=-1, verbose=3)(
        delayed(extract_features_safe)(fp, inst) for fp, inst in tasks
    )

    X = np.array([r[0] for r in results if r[0] is not None])
    y = np.array([r[1] for r in results if r[1] is not None])

    print(f"✅ Extracción completada en {time.time()-t0:.1f}s")
    print(f"   Total muestras válidas: {len(X)}")

    # Guardar caché para próximas ejecuciones
    os.makedirs(os.path.dirname(FEATURES_CACHE), exist_ok=True)
    np.savez(FEATURES_CACHE, X=X, y=y)
    print(f"   Caché guardado en: {FEATURES_CACHE}")

    return X, y


# ── Visualizaciones ───────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, class_names: list, save_path="metrics/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title("Matriz de Confusión — Random Forest IRMAS\n(valores normalizados por fila)", fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Matriz de confusión guardada en: {save_path}")


def plot_feature_importance(clf, save_path="metrics/feature_importance.png"):
    importances = clf.feature_importances_
    top_n = 20
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), importances[indices], color='steelblue', alpha=0.8)
    ax.set_title(f"Top {top_n} Features más importantes — Random Forest")
    ax.set_xlabel("Índice de feature")
    ax.set_ylabel("Importancia (Gini)")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(indices, rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Importancia de features guardada en: {save_path}")


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train_model(use_random_search: bool = False):
    print("=== ENTRENANDO MODELO IRMAS ===\n")

    X, y = load_irmas_dataset()
    print(f"\nTotal muestras: {len(X)} | Features por muestra: {X.shape[1]}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

    print("\nAplicando SMOTE para balancear clases...")
    print("Distribución antes de SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(le.classes_[unique], counts):
        print(f"  {u}: {c}")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"\nDistribución después de SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(le.classes_[unique], counts):
        print(f"  {u}: {c}")
    print(f"Total muestras train: {len(X_train)}")

    if use_random_search:
        # Búsqueda automática de hiperparámetros
        # Solo usar si el entrenamiento base fue rápido (<5 min)
        from sklearn.model_selection import RandomizedSearchCV

        print("\n🔍 Iniciando RandomizedSearchCV...")
        param_dist = {
            'n_estimators':      [200, 300, 400, 500, 600],
            'max_depth':         [15, 20, 25, 30, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf':  [1, 2, 3],
            'max_features':      ['sqrt', 'log2', 0.3, 0.4],
            'criterion':         ['gini', 'entropy'],
        }

        base_clf = RandomForestClassifier(n_jobs=-1, random_state=42)
        search = RandomizedSearchCV(
            base_clf,
            param_distributions=param_dist,
            n_iter=50,          # 50 combinaciones aleatorias
            cv=5,               # 5-fold CV para ser rápido
            scoring='f1_macro',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        t0 = time.time()
        search.fit(X_train, y_train)
        print(f"\n✅ RandomizedSearchCV completado en {time.time()-t0:.1f}s")
        print(f"   Mejores parámetros: {search.best_params_}")
        print(f"   Mejor F1 (CV):      {search.best_score_:.4f}")
        clf = search.best_estimator_

    else:
        # Entrenamiento directo con hiperparámetros base
        print("\nEntrenando Random Forest...")
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X_train, y_train)
        print(f"✅ Entrenamiento completado en {time.time()-t0:.1f}s")

    # Evaluación
    y_pred = clf.predict(X_test)
    class_names = le.classes_

    print("\n=== RESULTADOS EN TEST SET ===")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Guardar visualizaciones
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_feature_importance(clf)

    # Guardar modelo
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)

    print(f"\n✅ Modelo guardado en: {MODEL_PATH}")
    return clf, le


# ── Carga e inferencia ────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return clf, le


def predict_instruments(audio_path: str, top_k: int = 11) -> list[dict]:
    clf, le = load_model()

    y_audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    total_duration = librosa.get_duration(y=y_audio, sr=sr)

    segment_samples = 3 * sr
    n_segments = max(1, int(total_duration // 3))
    n_segments = min(n_segments, 20)

    all_probas = []

    for i in range(n_segments):
        start   = i * segment_samples
        end     = min(start + segment_samples, len(y_audio))
        segment = y_audio[start:end]

        if len(segment) < sr:
            continue

        import tempfile
        import soundfile as sf

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            # Escribir FUERA del with para que el handle esté cerrado
            sf.write(tmp_path, segment, sr)
            features = extract_features(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass  # si falla en Windows, ignorar — el SO lo limpiará

        proba = clf.predict_proba([features])[0]
        all_probas.append(proba)

    if not all_probas:
        return []

    mean_probas = np.mean(all_probas, axis=0)

    results = [
        {
            "label":    le.classes_[idx],
            "label_es": INSTRUMENT_NAMES.get(le.classes_[idx], le.classes_[idx]),
            "score":    float(prob)
        }
        for idx, prob in enumerate(mean_probas)
    ]

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model(use_random_search=True)