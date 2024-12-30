import os
import re
import tomllib
from pathlib import Path
from typing import List

import pymupdf
import torch
import typer
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_settings(file_path: str = "settings.toml") -> dict:
    with open(file_path, "rb") as f:
        settings = tomllib.load(f)
    return settings


def extract_file_paths(directory: str) -> List[Path]:
    dir_path = Path(directory)
    file_paths = [dir_path / file_name for file_name in os.listdir(dir_path)]
    return file_paths


def extract_text(file_path: Path) -> str:
    try:
        doc = pymupdf.open(file_path)
        first_page_text = doc[0].get_text()
        return first_page_text
    except Exception as e:
        logger.error(f"Error during PDF opening or text extraction: {e}")
        return ""


def extract_paper_info(
    extracted_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    settings: dict,
) -> str:
    messages = [
        {"role": "system", "content": settings["system_prompt"]},
        {"role": "user", "content": settings["user_prompt"] + extracted_text},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128, # increase max_new_tokens
            temperature=0.1,
            do_sample=True,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        cleaned_response = clean_model_output(response)
        return cleaned_response
    except Exception as e:
        logger.error(f"Error during model response: {e}")
        return ""


def clean_pdf_text(text: str) -> str:
    text = re.sub(r"[-*∗:†]", " ", text).strip()
    return text


def clean_model_output(text: str) -> str:
    text = re.sub(r'[\\/*?"<>†|\[\]]', "-", text)
    text = re.sub(r"[@.:]", "", text)
    text = re.sub(r"[\r\n]+", "-", text)
    text = re.sub(r"^-+$", "", text)
    text = re.sub(r"  ", " ", text)
    text = text.strip("- ")
    return text


def main(directory: str) -> None:
    settings = load_settings()
    model_name = settings["model_name"].split("/")[-1]
    logger.info(f"Model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}:{torch.cuda.get_device_name()}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            settings["model_name"],
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(settings["model_name"])
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        return

    file_paths = extract_file_paths(Path(directory))
    file_count = len(file_paths)
    if file_count == 0:
        logger.warning(f"No files found in {directory}!")
        return

    file_suffix = "s" if file_count > 1 else ""
    logger.info(f"Found {file_count} file {file_suffix} in {directory}.")
    for file_path in file_paths:
        try:
            extracted_text = extract_text(file_path)
            extracted_text = clean_pdf_text(extracted_text)
            if not extracted_text:
                logger.warning(f"No text extracted from {file_path.name}. Skipping file.")
                continue
                
            extracted_info = extract_paper_info(extracted_text, model, tokenizer, settings)
            if not extracted_info:
                logger.warning(f"Failed to extract paper info from {file_path.name}. Leaving it un-renamed.")
                continue
                
            new_name = f"{extracted_info}{file_path.suffix}"
            os.rename(file_path, Path(file_path.parent, new_name))
            logger.info(f"{file_path.name} renamed to {new_name}.")

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}. Error: {e}")

    logger.info("Finished.")


if __name__ == "__main__":
    typer.run(main)