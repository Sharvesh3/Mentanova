from fastapi import FastAPI, HTTPException
import google.generativeai as genai
import os

app = FastAPI(title="Gemini Test API", version="1.0.0")

def configure_gemini():
    """
    Configure the Gemini API with the environment key.
    Raises RuntimeError if the key is missing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required!")
    genai.configure(api_key=api_key)

@app.get("/")
async def root():
    return {"message": "Gemini API Test Running"}

@app.get("/gemini/models")
async def list_models():
    """
    List all available Gemini models that support content generation.
    """
    try:
        configure_gemini()
        models_list = []

        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                models_list.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description[:200] if model.description else "N/A",
                    "methods": model.supported_generation_methods
                })
        return {"models": models_list}

    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error fetching models: {str(e)[:200]}")

@app.get("/gemini/test")
async def test_models():
    """
    Test common Gemini models with a simple "Hello" prompt.
    """
    try:
        configure_gemini()

        test_models = [
            "gemini-pro",
            "models/gemini-pro",
            "gemini-1.5-flash-8b",
            "models/gemini-1.5-flash-8b",
            "gemini-2.0-flash-exp",
            "models/gemini-2.0-flash-exp",
        ]

        results = []

        for model_name in test_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello")
                results.append({
                    "model": model_name,
                    "status": "WORKS",
                    "response": response.text[:200]
                })
            except Exception as e:
                error_msg = str(e)[:200]
                results.append({
                    "model": model_name,
                    "status": "ERROR",
                    "error": error_msg
                })

        return {"results": results}

    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error testing models: {str(e)[:200]}")
