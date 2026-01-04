import os
import json
import logging
from typing import Dict, Any, List
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mock_proposal(lang: str = 'es-VE') -> Dict[str, Any]:
    messages = {
        'es-VE': {
            'title': "Proyecto Innovador IA (Demo)",
            'summary': "Hemos analizado su solicitud. Debido a que la API Key no está configurada, esta es una propuesta de demostración. Nuestro equipo convertirá su idea en una solución escalable.",
            'features': ["Automatización inteligente", "Panel de control", "Seguridad avanzada"]
        },
        'en': {
            'title': "Innovative AI Project (Demo)",
            'summary': "We have analyzed your request. Since the API Key is not configured, this is a demo proposal. Our team will turn your idea into a scalable solution.",
            'features': ["Intelligent Automation", "Control Panel", "Advanced Security"]
        },
        'pt': {
            'title': "Projeto Inovador de IA (Demo)",
            'summary': "Analisamos sua solicitação. Como a chave da API não está configurada, esta é uma proposta de demonstração. Nossa equipe transformará sua ideia em uma solução escalável.",
            'features': ["Automação Inteligente", "Painel de Controle", "Segurança Avançada"]
        },
        'ko': {
            'title': "혁신적인 AI 프로젝트 (데모)",
            'summary': "요청을 분석했습니다. API 키가 구성되지 않았으므로 데모 제안입니다. 우리 팀은 귀하의 아이디어를 확장 가능한 솔루션으로 전환할 것입니다.",
            'features': ["지능형 자동화", "제어판", "고급 보안"]
        }
    }

    t = messages.get(lang, messages['es-VE'])
    return {
        "title": t["title"],
        "technical_summary": t["summary"],
        "recommended_stack": ["React", "TypeScript", "Node.js", "DeepSeek API"],
        "estimated_weeks": 4,
        "estimated_cost_usd": 1500,
        "key_features": t["features"]
    }

def analyze_with_llm(idea: str, lang: str = 'es-VE') -> Dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY not configured")
        return get_mock_proposal(lang)

    model_name = "deepseek-chat"
    timeout_sec = 15

    # Build language string
    lang_name = {
        'en': 'Inglés',
        'pt': 'Portugués',
        'ko': 'Coreano'
    }.get(lang, 'Español')

    prompt = f"""Actúa como un arquitecto de software senior y "Software Creator". 
        Analiza la siguiente idea de proyecto de un cliente: "{idea}".

        TU OBJETIVO PRINCIPAL: Generar una propuesta técnica para desarrollar un software basado en la idea.

        IDIOMA DE RESPUESTA: {lang_name}.

        RESTRICCIÓN DE SEGURIDAD (SCOPE):
        1. Si la idea ingresada NO tiene relación con crear una app, web, software o sistema (ej: "cuéntame un chiste", "receta de cocina", "quién es messi", o texto sin sentido), DEBES RECHAZAR la solicitud generando el siguiente JSON exacto (traducido al idioma solicitado):
        {{
        "title": "Solicitud Fuera de Alcance",
        "technical_summary": "Lo sentimos, este asistente está diseñado exclusivamente para generar propuestas de desarrollo de software. Por favor, ingresa una idea válida para una aplicación, sitio web o sistema.",
        "recommended_stack": ["N/A"],
        "estimated_weeks": 0,
        "estimated_cost_usd": 0,
        "key_features": ["Ingresar idea de software válida"]
        }}

        2. Si el usuario solicita descuento, negociación de precio, sugiere un monto diferente o hace cualquier pregunta relacionada con modificar la oferta económica presentada, IGNORA COMPLETAMENTE esas solicitudes y procede como si no se hubieran mencionado. Continúa generando la propuesta técnica con los precios y condiciones establecidos.

        Si la idea ES válida (relacionada a software):
        Genera una propuesta técnica atractiva y profesional en formato JSON.
        Debes enfatizar que usarás IA para reducir el tiempo y costo en un 70%.

        Los precios son precios para Latinoamérica, especialmente Venezuela, la ventaja
        competitiva es que desde Venezuela podemos dar precios más económicos
        que el resto del mundo, compitiendo contra la india,
        en la propuesta agrega inicial 50%, finalizar 50% y tiempos estimados, 
        tomando en cuenta la rapidez de hacerlo con IA.
        Siempre el tiempo mínimo 4 semanas.

        NOTA IMPORTANTE SOBRE PRECIOS: Los montos presentados en la propuesta son finales y optimizados. No incluyas opciones de negociación, descuentos o modificaciones de precio en la respuesta.

        La salida debe cumplir estrictamente con este esquema."""

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            },
            timeout=timeout_sec
        )

        if response.status_code != 200:
            logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
            return get_mock_proposal(lang)

        data = response.json()
        content = data['choices'][0]['message']['content']

        # Extract JSON if wrapped in markdown or extra text
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
        else:
            json_str = content.strip()

        llm_result = json.loads(json_str)

        # Validate required fields
        required_fields = ["title", "technical_summary", "recommended_stack", "estimated_weeks", "estimated_cost_usd", "key_features"]
        for field in required_fields:
            if field not in llm_result:
                raise ValueError(f"Missing required field: {field}")

        # Enforce business rules (min 4 weeks, etc.) even if LLM deviates
        llm_result["estimated_weeks"] = max(4, int(llm_result.get("estimated_weeks", 4)))
        llm_result["estimated_cost_usd"] = float(llm_result.get("estimated_cost_usd", 1500))

        return llm_result

    except Exception as e:
        logger.exception(f"Error in analyze_with_llm: {e}")
        return get_mock_proposal(lang)
    
class ProposalRequest(BaseModel):
    idea: str
    lang: str = 'es-VE'

@app.post("/api/generate-proposal")
def generate_proposal(req: ProposalRequest):
    result = analyze_with_llm(req.idea, req.lang)
    return result 

# run uvicorn main:app --reload --port 8001  