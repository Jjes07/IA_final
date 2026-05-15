import gradio as gr
import warnings
import os
warnings.filterwarnings("ignore")

from model import (
    analyze_audio,
    build_bar_chart,
    build_timeline_chart,
    format_results_for_bot
)
from bot import chat_with_bot


# ── Funciones auxiliares ──────────────────────────────────────────────────────

def update_filename(audio_path):
    if audio_path is None:
        return "<p style='text-align:center; color:#666; font-size:13px; margin:4px 0'>Sin archivo cargado</p>"
    name = os.path.basename(audio_path)
    return f"<p style='text-align:center; color:#4f46e5; font-weight:bold; font-size:14px; margin:4px 0'>🎵 {name}</p>"


def clear_results():
    empty_html = "<p style='text-align:center; color:#666; font-size:13px; margin:4px 0'>Sin archivo cargado</p>"
    return None, None, "", "", empty_html


def bot_response(message, history, bot_context):
    if not message or (isinstance(message, str) and not message.strip()):
        return history, ""
    response = chat_with_bot(message, history, bot_context)
    history = history + [
        {"role": "user",      "content": message if isinstance(message, str) else str(message)},
        {"role": "assistant", "content": response},
    ]
    return history, ""


# ── Análisis con AST ──────────────────────────────────────────────────────────

def run_analysis_ast(audio_path):
    if audio_path is None:
        yield None, None, "⚠️ Por favor sube un archivo de audio primero.", ""
        return
    try:
        filename = os.path.basename(audio_path)
        yield None, None, f"⏳ [AST] Analizando '{filename}'... esto puede tomar unos segundos.", ""

        results = analyze_audio(audio_path)

        if not results["global_results"]:
            yield None, None, "No se detectaron instrumentos con suficiente confianza. Intenta con otra canción.", ""
            return

        bar_chart      = build_bar_chart(results["global_results"])
        timeline_chart = build_timeline_chart(results["timeline"], results["duration"])
        bot_context    = format_results_for_bot(results)

        top3 = list(results["global_results"].items())[:3]
        top3_str = ", ".join(f"{k} ({v*100:.0f}%)" for k, v in top3)
        summary = (
            f"✅ [AST] '{filename}' — {results['duration']:.1f}s, "
            f"{results['n_segments']} segmentos analizados.\n"
            f"🎵 Instrumentos principales: {top3_str}\n"
            f"ℹ️ Los % indican presencia relativa entre instrumentos detectados."
        )

        yield bar_chart, timeline_chart, summary, bot_context

    except Exception as e:
        yield None, None, f"❌ Error AST: {str(e)}", ""


# ── Análisis con Random Forest (IRMAS) ───────────────────────────────────────

def run_analysis_rf(audio_path):
    if audio_path is None:
        yield None, None, "⚠️ Por favor sube un archivo de audio primero.", ""
        return
    try:
        from model_irmas import predict_instruments, INSTRUMENT_NAMES

        filename = os.path.basename(audio_path)
        yield None, None, f"⏳ [Random Forest] Analizando '{filename}'...", ""

        results_list = predict_instruments(audio_path, top_k=11)

        if not results_list:
            yield None, None, "No se detectaron instrumentos.", ""
            return

        # Convertir al formato que espera build_bar_chart
        global_results = {
            r["label_es"]: r["score"]
            for r in results_list
            if r["score"] > 0.03
        }

        bar_chart = build_bar_chart(global_results)

        # RF no genera timeline temporal — se deja vacío
        timeline_chart = None

        # Contexto para el bot
        lines = [
            f"Modelo: Random Forest entrenado en IRMAS (11 instrumentos)",
            f"Archivo analizado: {filename}",
            "",
            "Instrumentos detectados (probabilidad promedio):"
        ]
        for r in results_list:
            lines.append(f"  - {r['label_es']}: {r['score']*100:.1f}%")
        bot_context = "\n".join(lines)

        top3_str = ", ".join(
            f"{r['label_es']} ({r['score']*100:.0f}%)"
            for r in results_list[:3]
        )
        summary = (
            f"✅ [Random Forest] '{filename}' analizado.\n"
            f"🎵 Instrumentos principales: {top3_str}\n"
            f"ℹ️ Modelo propio entrenado en IRMAS — clasifica entre 11 instrumentos."
        )

        yield bar_chart, timeline_chart, summary, bot_context

    except FileNotFoundError:
        yield None, None, (
            "❌ El modelo Random Forest no está entrenado aún.\n"
            "Ejecuta 'python model_irmas.py' en la terminal primero."
        ), ""
    except Exception as e:
        yield None, None, f"❌ Error RF: {str(e)}", ""


# ── Interfaz ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="🎵 Instrument Detector",
    theme=gr.themes.Soft(),
    css="""
        .main-title { text-align: center; margin-bottom: 8px; }
        .subtitle   { text-align: center; color: #666; margin-bottom: 20px; }
        #chatbot-col { border-left: 1px solid #e5e7eb; padding-left: 16px; }
    """
) as demo:

    bot_context_state = gr.State("")

    gr.HTML("<h1 class='main-title'>🎸 Instrument Detector</h1>")
    gr.HTML("<p class='subtitle'>Identifica los instrumentos musicales presentes en cualquier canción usando IA</p>")

    with gr.Row():

        # ── Columna izquierda: análisis ───────────────────────────────────
        with gr.Column(scale=3):

            filename_display = gr.HTML(
                "<p style='text-align:center; color:#666; font-size:13px; margin:4px 0'>Sin archivo cargado</p>"
            )

            audio_input = gr.Audio(
                label="🎵 Canción — sube un archivo o elige un ejemplo",
                type="filepath",
                sources=["upload"],
            )

            gr.Examples(
                examples=[
                    ["assets/demo_songs/Blackbird.mp3"],
                    ["assets/demo_songs/Hotel California.mp3"],
                    ["assets/demo_songs/Careless Whisper.mp3"],
                    ["assets/demo_songs/Fur Elise.mp3"],
                    ["assets/demo_songs/Seven Nation Army.mp3"],
                ],
                inputs=[audio_input],
                label="🎧 Ejemplos de demo"
            )

            # Dos botones de análisis
            with gr.Row():
                analyze_ast_btn = gr.Button(
                    "🔍 Analizar con AST",
                    variant="primary",
                    scale=1
                )
                analyze_rf_btn = gr.Button(
                    "🌲 Analizar con Random Forest",
                    variant="secondary",
                    scale=1
                )

            status_text = gr.Textbox(
                label="Estado",
                interactive=False,
                lines=3,
                placeholder="El resultado del análisis aparecerá aquí..."
            )

            with gr.Tabs():
                with gr.TabItem("📊 Instrumentos detectados"):
                    bar_chart_output = gr.Image(show_label=False)
                with gr.TabItem("⏱️ Línea de tiempo"):
                    timeline_output = gr.Image(show_label=False)

        # ── Columna derecha: bot ──────────────────────────────────────────
        with gr.Column(scale=2, elem_id="chatbot-col"):
            gr.HTML("<h3 style='margin-top:0'>🤖 MusicBot</h3>")
            gr.HTML("<p style='color:#666; font-size:13px'>Pregúntame sobre los resultados o cómo funciona el sistema.</p>")

            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "¡Hola! Soy MusicBot 🎵 Sube una canción y analízala — luego puedo explicarte qué instrumentos detecté y por qué."}],
                height=480,
                show_label=False,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Escribe tu pregunta...",
                    show_label=False,
                    scale=5,
                    lines=1
                )
                send_btn = gr.Button("Enviar", scale=1, variant="secondary")

    # ── Eventos ───────────────────────────────────────────────────────────────

    # Actualizar nombre del archivo y limpiar resultados al cambiar audio
    audio_input.change(
        fn=clear_results,
        inputs=[],
        outputs=[bar_chart_output, timeline_output, status_text, bot_context_state, filename_display]
    )

    audio_input.change(
        fn=update_filename,
        inputs=[audio_input],
        outputs=[filename_display]
    )

    # Botón AST
    analyze_ast_btn.click(
        fn=run_analysis_ast,
        inputs=[audio_input],
        outputs=[bar_chart_output, timeline_output, status_text, bot_context_state]
    )

    # Botón Random Forest
    analyze_rf_btn.click(
        fn=run_analysis_rf,
        inputs=[audio_input],
        outputs=[bar_chart_output, timeline_output, status_text, bot_context_state]
    )

    # Bot — Enter
    msg_input.submit(
        fn=bot_response,
        inputs=[msg_input, chatbot, bot_context_state],
        outputs=[chatbot, msg_input]
    )

    # Bot — botón Enviar
    send_btn.click(
        fn=bot_response,
        inputs=[msg_input, chatbot, bot_context_state],
        outputs=[chatbot, msg_input]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=8000,
        show_error=True
    )