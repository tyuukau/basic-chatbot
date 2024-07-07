import torch

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline


def get_huggingface_llm(
    model_name: str = "lmsys/vicuna-7b-v1.5", max_new_token: int = 512
) -> HuggingFacePipeline:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=nf4_config, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
    )
    return llm


LLM = get_huggingface_llm()
