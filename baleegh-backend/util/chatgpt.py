from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["CHATGPT_API"],
)

def chatgpt_translation(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are a professional translator, your task is to translate from English to Classic Arabic.\n"
                    "English: He felt lost in the crowd.\n"
                    "Literal Translation: شعر بأنه تائه في الحشد.\n"
                    "Classical Arabic Translation: غاب في زحام الناس.\n\n"
                        
                    "English: He was troubled by the unexpected news.\n"
                    "Literal Translation: كان منزعجًا من الخبر غير المتوقع.\n"
                    "Classical Arabic Translation: أثقلته المفاجأة.\n\n"

                    "English: He lost hope completely.\n"
                    "Literal Translation: فقد الأمل تمامًا.\n"
                    "Classical Arabic Translation: خاب أمله كل الخيبة.\n\n"

                    "English: They turned a blind eye to his mistakes.\n"
                    "Literal Translation: غضوا النظر عن أخطائه.\n"
                    "Classical Arabic Translation: تجاهلوا زلاته.\n\n"

                    "English: He felt uncomfortable in their presence.\n"
                    "Literal Translation: شعر بعدم الارتياح في وجودهم.\n"
                    "Classical Arabic Translation: ضاق بهم.\n\n"

                    "English: They were quick to judge without understanding.\n"
                    "Literal Translation: كانوا سريعين في الحكم دون فهم.\n"
                    "Classical Arabic Translation: تسرعوا في الحكم بلا دراية.\n\n"

                    "English: He couldn’t sleep from worry.\n"
                    "Literal Translation: لم يستطع النوم من القلق.\n"
                    "Classical Arabic Translation: أرقه القلق.\n\n"

                    "These are examples of translations from English to true Arabic and literal Arabic. I want you to translate to the true Arabic style.\n"
                )

            },
            {
                "role": "user", 
                "content": (f"English: {prompt}\n"
                            "Classical Arabic Transltion:")
            }
        ]
    )
    return response.choices[0].message.content.strip()