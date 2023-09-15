import argparse
import json
import re
import os

from tqdm import tqdm

def contains_unwanted_words(text):
    unwanted_words = [
        "Do you have any other questions",
        "Is there anything else",
        "As an AI assistant",
        "Can I assist you with anything else",
        "Do you need any",
        "clarify your request",
        "Could you please provide more",
        "me to help you with?",
        "anything specific you",
		"text-based AI language model",
		"it is never okay",
		"as a language model",
		"as an AI language model",
		"As a large language model",
		"As an AI",
		"it is not appropriate",
		"it's not appropriate",
		"I cannot fulfill your request",
		"cannot provide guidance",
		"cannot provide information",
		"unable to offer assistance",
		"cannot engage in discussions",
		"programming prohibits",
		"cannot support or promote",
		"against my programming",
		"not able to provide",
		"cannot provide any information",
		"an AI language model you don't have",
		"As an AI language model, I cannot",
		"As an AI language model, I do not",
		"As an AI language model, I am not able",
		"As an AI language model, I don't have personal",
		"I am an AI language model and do not",
		"However, it is important to use any code or information provided responsibly and within legal and ethical boundaries.",
		"As an AI language model, I don't have",
		"As an AI language model, I am only able",
		"AI language model and I do not",
		"As an AI language model, I cannot modify",
		"As an AI language model, I do not",
		"I know as an AI language model you don't have",
		"as an AI language model, you cannot",
		"I'm sorry, but as an AI language model",
		"As an AI language model, I don't have",
		"Unfortunately, I cannot provide",
		"I'm sorry, I cannot",
		"I'm sorry, I cannot generate",
		"AI cannot create or program",
		"I'm afraid I cannot create",
		"you cannot create an",
		"Lo siento"
		"como modelo de lenguaje AI",
		"Lo siento, como modelo de lenguaje",
		"no puedo proporcionar",
		"pero debido a mi capacidad para generar c\u00f3digos complejos y completos es limitado",
		"Lo siento, pero no puedo",
		"Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
		"Lo siento, como modelo de lenguaje, no tengo",
		"Lo siento, debe haber habido una confusi\u00f3n",
		"Lo siento, como modelo de lenguaje, no puedo realizar",
		"Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
		"Lamento no poder proporcionarte el c\u00f3digo",
		"Desculpe-me, mas a linguagem vulgar e ofensiva",
		"apropriada em nenhum contexto",
		"Como modelo de linguagem",
		"Como um modelo de linguagem, n\u00e3o tenho a capacidade de",
		"I cannot assist",
		"I'm sorry,",
		"I'm an",
		"I am an",
		"I'm an AI" ,
		"I am an AI",
		"I cannot provide",
		"I can't provide",
		"I won't provide",
		"I cannot",
		"As a machine",
		"I don't have the ability",
		"I am here to assist",
		"my purpose is to ",
		"my knowledge cutoff",
		"my knowledge cut off",
		"September 2021",
		"I apologize, but",
		"It is not possible",
		"It's not possible",
		"It is impossible",
		"It's impossible"
	]
    for word in unwanted_words:
        if word.lower() in text.lower():
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    in_file = ''
    out_file = ''    

    content = json.load(open(in_file, "r"))
    num_conv = len(content)

    new_content_id = []
    new_content = []
    for i in tqdm(range(len(content))):
        if not contains_unwanted_words(content[i]["output"]):
            new_content.append(content[i])
            new_content_id.append(i)

    print(f"return {len(new_content_id)} out of {len(content)}, start dump ...")
    json.dump(new_content, open(out_file, "w"), indent=4)
    

    pass