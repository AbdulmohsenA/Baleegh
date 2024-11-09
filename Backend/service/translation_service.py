from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
import os
import re
from util.allam_model import AllamModel
import torch
import re
import weave
from dotenv import load_dotenv

load_dotenv()

class TranslationService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Abdulmohsena/Faseeh")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Abdulmohsena/Faseeh").to("cuda")
        self.generation_config = GenerationConfig.from_pretrained("Abdulmohsena/Faseeh")
        self.model.eval()

    @weave.op
    def translate(self, text):
        encoded_ar = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated_tokens = self.model.generate(**encoded_ar, generation_config=self.generation_config)
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    @weave.op
    def allam(self, query):
        model = AllamModel(
            model_id=os.environ["IBM_MODEL_ID"], 
            project_id=os.environ["IBM_PROJECT_ID"]
        )
        return model.generate_text(query)
    
    def preprocess_response(self, text):
        return re.sub(r'[A-Za-z:]', '', text)
