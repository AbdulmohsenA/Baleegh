from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import modal
import logging
import os
from dotenv import load_dotenv
from util.modal_image import get_image
import weave
import wandb
from service.translation_service import TranslationService

load_dotenv()

MODEL_DIR = "BALEEGH"
MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App("baleegh", image=get_image(), secrets=[modal.Secret.from_name("env-variables")])

log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(level=log_level)

@app.cls(container_idle_timeout=5 * MINUTES, timeout=24 * HOURS, keep_warm=1, gpu='t4')
class WebApp:
    def __init__(self):
        # Initialize FastAPI app
        self.web_app = FastAPI()
        
        # Set up CORS
        self.web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Adjust as needed
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.web_app.add_api_route("/", self.query)

    @modal.build()  # add another step to the image build
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download

        os.makedirs(MODEL_DIR, exist_ok=True)
        snapshot_download("Abdulmohsena/Faseeh", local_dir=MODEL_DIR)
    
    @modal.enter()
    def setup(self):
        wandb.login(key=os.environ['WANDB_KEY'])
        weave.init("Baleegh")
        self.translation_service = TranslationService()

    def query(self, text: str):
        model_response = self.translation_service.translate(text)
        allam_response = self.translation_service.allam(model_response)
        result = self.translation_service.preprocess_response(allam_response)
        return JSONResponse(content={"translation": result})
    
    @modal.asgi_app()
    def fastapi_app(self):
        return self.web_app