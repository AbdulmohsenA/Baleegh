import os
import requests
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials

load_dotenv()

def get_ibm_access_token(api_key):
    url = 'https://iam.cloud.ibm.com/identity/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
        'apikey': api_key
    }

    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()  # Ensure we raise an error for bad responses
    return response.json().get('access_token')

class AllamModel:
    def __init__(self, model_id, project_id, url="https://eu-de.ml.cloud.ibm.com"):
        api_key = os.environ['IBM_API_KEY']
        if not api_key:
            raise ValueError("IBM_API_KEY environment variable not set")

        self.credentials = Credentials(
            url=url,
            token=get_ibm_access_token(api_key)
        )
        self.parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 100,
            "repetition_penalty": 1.0,
            "temperature": 0.0,
        }
        self.model = Model(
            model_id=model_id,
            credentials=self.credentials,
            params=self.parameters,
            project_id=project_id
        )
        self.system_prompt = "\n".join([
            "أنت مختص في اللغة العربية، ودورك هو تصحيح الأخطاء الإملائية والنحوية في الجمل العربية التي تتلقاها.",
            "يجب أن تقوم بالتغييرات الضرورية فقط، دون تغيير الكلمات أو المفردات إلا إذا كان ضروريًا للنحو.",
            "أجب فقط بالجملة المصححة دون إضافة أي شرح للتعديلات أو الأخطاء.",
            "مثال: إذا أعطيتك 'في نفس الوقت'، يجب أن ترد بـ'في الوقت نفسه' فقط، أو إذا أعطيتك 'لم أرى' ترد بـ'لم أرَ' فقط دون أي تعليق."
        ])
    
    def construct_prompt(self, query):
        prompt = f"{self.system_prompt}\n\nالجملة المراد تصحيحها: {query}\nالجملة الصحيحة:"
        return prompt
    
    def generate_text(self, query):
        prompt = self.construct_prompt(query)
        return self.model.generate_text(prompt)
