import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODELO = "claude-haiku-4-5-20251001"  # rápido y barato, perfecto para el bot

client = None

def init_client():
    global client
    if not ANTHROPIC_API_KEY:
        print("⚠️ ANTHROPIC_API_KEY no encontrada en .env — bot desactivado")
        return False
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("✅ Bot Claude inicializado")
        return True
    except Exception as e:
        print(f"❌ Error inicializando bot: {e}")
        return False


def build_system_prompt(analysis_context: str) -> str:
    return f"""Eres MusicBot, un asistente experto en análisis musical e identificación de instrumentos.

Este sistema usa DOS modelos de IA complementarios que el usuario puede elegir:

MODELO 1 — AST (Audio Spectrogram Transformer):
- Red neuronal profunda preentrenada en AudioSet de Google (2 millones de clips, 527 categorías)
- Detecta instrumentos en ventanas de 5 segundos y genera una línea de tiempo de presencia
- Los porcentajes indican presencia RELATIVA entre los instrumentos detectados (renormalizados)
- Cobertura amplia pero probabilidades bajas en valor absoluto por competir entre 527 categorías

MODELO 2 — Random Forest (entrenado en IRMAS):
- Modelo de aprendizaje de máquina clásico entrenado con 6,705 clips del dataset IRMAS
- Clasifica entre 11 instrumentos: Cello, Clarinete, Flauta, Guitarra acústica, Guitarra eléctrica, Órgano, Piano, Saxofón, Trompeta, Violín y Voz
- Features usadas: MFCCs, Chroma, Spectral Contrast, ZCR, RMS (141 dimensiones)
- Accuracy en test set: 62% | F1 macro: 0.62
- Limitación principal: entrenado con instrumentos solos (IRMAS), puede tener dificultades con mezclas reales
- No genera línea de tiempo — analiza segmentos de 3 segundos y promedia probabilidades

CONTEXTO DEL ANÁLISIS ACTUAL:
{analysis_context}

TUS CAPACIDADES:
1. Explicar qué instrumentos fueron detectados y con qué confianza
2. Explicar la diferencia entre los dos modelos y cuándo usar cada uno
3. Describir las características sonoras de cada instrumento detectado
4. Explicar por qué ciertos instrumentos son más difíciles de detectar
5. Dar contexto musical sobre la canción o género analizado
6. Explicar cómo funcionan el AST y el Random Forest internamente
7. Guiar al usuario en el uso de la interfaz

INSTRUCCIONES:
- Si aún no se ha analizado ningún audio, invita al usuario a subir una canción y elegir un modelo
- Si hay resultados disponibles, úsalos como base de tus respuestas
- Cuando el contexto indique "Modelo: Random Forest", explica los resultados considerando sus limitaciones
- Cuando el contexto indique análisis AST, explica que los % son relativos entre instrumentos detectados
- Sé conciso pero informativo — respuestas de máximo 3 párrafos
- Usa lenguaje accesible, no excesivamente técnico
"""


def chat_with_bot(message, history: list, analysis_context: str) -> str:
    if client is None:
        return "⚠️ El bot no está disponible. Verifica tu ANTHROPIC_API_KEY en el archivo .env"

    # Normalizar message a string
    if isinstance(message, list):
        message = " ".join(str(m) for m in message)
    elif not isinstance(message, str):
        message = str(message)
    message = message.strip()

    if not message:
        return "Por favor escribe una pregunta."

    try:
        system_prompt = build_system_prompt(analysis_context)

        # Construir historial en formato Anthropic
        messages = []
        for item in history:
            if isinstance(item, dict):
                role = "assistant" if item.get("role") == "assistant" else "user"
                content = item.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                content = clean_content(str(content)).strip()
                if content:
                    messages.append({"role": role, "content": content})

        # Agregar mensaje actual
        messages.append({"role": "user", "content": message})

        response = client.messages.create(
            model=MODELO,
            max_tokens=512,
            system=system_prompt,
            messages=messages
        )

        block = response.content[0]
        return block.text if hasattr(block, 'text') else str(block)

    except Exception as e:
        return f"Error en el bot: {str(e)}"

def clean_content(content):
    """Limpia contenido que llegó como repr de objeto en lugar de texto."""
    if not isinstance(content, str):
        return str(content)
    # Si el string parece un dict serializado, extraer el texto
    if content.startswith("{'text':") or content.startswith('{"text":'):
        try:
            import ast
            parsed = ast.literal_eval(content)
            if isinstance(parsed, dict) and 'text' in parsed:
                return parsed['text']
        except:
            pass
    return content

init_client()