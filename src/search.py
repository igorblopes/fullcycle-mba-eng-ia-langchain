import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()


for k in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if k not in os.environ:
        raise RuntimeError(f"Environment variable {k} not set")
    
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question=None):
  embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
  store = PGVector(
      embeddings=embeddings,
      collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
      connection=os.getenv("DATABASE_URL"),
      use_jsonb=True,
  )

  results = store.similarity_search_with_score(question, k=10)
  
  base_results = format_results(results=results)

  prompt = PROMPT_TEMPLATE.format(contexto=base_results, pergunta=question)

  response_prompt = request_gpt(prompt=prompt)

  return response_prompt

def format_results(results):
  output = []

  for i, (doc, score) in enumerate(results, start=1):
    output.append("-" * 50)
    output.append(f"Result {i} (score: {score:.4f})")
    output.append("-" * 50)

    output.append("\nTexto:\n")
    output.append(doc.page_content.strip())

    output.append("\nMetadados:\n")
    for k, v in doc.metadata.items():
      output.append(f"{k}: {v}")

    output.append("\n") 

  return "\n".join(output)

def request_gpt(prompt):
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

  response = client.chat.completions.create(
      model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano"),
      messages=[
          {"role": "user", "content": prompt},
      ],
      temperature=1
  )

  return response.choices[0].message.content.strip()

