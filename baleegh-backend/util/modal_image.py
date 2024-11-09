import modal
def get_image():
    return modal.Image.debian_slim().pip_install(
        "fastapi==0.115.2",
        "ibm_watsonx_ai==1.1.15",  
        "python-dotenv==1.0.1",
        "weave==0.51.17",
        "wandb==0.18.5",
        "openai",
        "huggingface",
        "huggingface-hub",
        "transformers",
        "torch"
    )