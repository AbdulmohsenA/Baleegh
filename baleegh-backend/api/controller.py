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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from util.allam_model import AllamModel
import torch
import re


load_dotenv()



MODEL_DIR = "BALEEGH"
MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App("baleegh", image=get_image(), secrets=[modal.Secret.from_name("env-variables")])

log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(level=log_level)

@app.cls(container_idle_timeout=5 * MINUTES, timeout=24 * HOURS, keep_warm=3)
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
        self.tokenizer = AutoTokenizer.from_pretrained("Abdulmohsena/Faseeh")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Abdulmohsena/Faseeh")
        self.generation_config = GenerationConfig.from_pretrained("Abdulmohsena/Faseeh")

    @weave.op
    def model_translation(self, text):
        encoded_ar = self.tokenizer(text, return_tensors="pt")
        self.model.eval()
        with torch.inference_mode():
            generated_tokens = self.model.generate(**encoded_ar, generation_config=self.generation_config)
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    def query(self, text: str):
        model_response = self.model_translation(text)
        allam_response = self.allam(model_response)
        result = self.preprocess_model_response(allam_response)
        return JSONResponse(content={"translation": result})
    
    def preprocess_model_response(self, text):
        result = re.sub(r'[A-Za-z:]', '', text)
        return result
    
    @weave.op
    def allam(self, query):
        model = AllamModel(
            model_id=os.environ["IBM_MODEL_ID"], 
            project_id=os.environ["IBM_PROJECT_ID"]
        )
        return model.generate_text(query)
    
    @modal.asgi_app()
    def fastapi_app(self):
        return self.web_app

