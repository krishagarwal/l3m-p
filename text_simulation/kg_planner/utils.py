from __future__ import annotations
import os
from pydantic import BaseModel
import psycopg2
from psycopg2 import sql

from llama_index.core.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.core.llms.llm import LLM


def reset_database(dbname, user, password, host, port, schema_file):
    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(dbname='postgres', user=user, password=password, host=host, port=port)
        conn.autocommit = True  # Ensure we can execute CREATE/DROP DATABASE statements
        cur = conn.cursor()

        # Optionally drop and recreate the database (useful for full reset)
        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {};").format(sql.Identifier(dbname)))
        cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(dbname)))
        conn.commit()

        # Now reconnect to the newly created database (if dropped and recreated)
        cur.close()
        conn.close()
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        cur = conn.cursor()

        # Open and execute the schema file
        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Execute the schema SQL
        cur.execute(schema_sql)
        conn.commit()
        print("Database has been reset and schema executed successfully.")

    except Exception as error:
        print(f"Error: {error}")

    finally:
        if conn:
            cur.close()
            conn.close()


# read prompt template from file and format
def get_prompt_template(filename: str, **kwargs) -> str:
    with open(os.path.join(os.path.dirname(__file__), filename), "r") as f:
        contents = f.read()
    if not kwargs:
        return contents
    return contents.format(**kwargs)

class EntitySelection(BaseModel):
    KEYWORDS: list[str]

def extract_keywords(llm: LLM,
                     template: str,
                     query_str: str,
                     max_keywords: int = 10,
                     always_include: list[str] = []) -> list:
    ENTITY_SELECT_PROMPT = PromptTemplate(
        template,
        prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
    )
    response = llm.as_structured_llm(EntitySelection).predict( #type: ignore
        ENTITY_SELECT_PROMPT,
        max_keywords=max_keywords,
        question=query_str,
    )
    extracted_entities = set(EntitySelection.model_validate_json(response).KEYWORDS)
    extracted_entities.update(always_include)
    return extracted_entities # type: ignore
