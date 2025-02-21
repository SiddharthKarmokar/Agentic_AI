import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.model.groq import Groq

import os
from dotenv import load_dotenv
load_dotenv()


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = PDFKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    path="downloads/",
    vector_db=PgVector2(collection="receipes", db_url=db_url)
)

knowledge_base.load()

storage=PgAssistantStorage(table_name="pdf_assitant", db_url=db_url)


def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]
    
    assistant = Assistant(
        model=Groq(id="llama-3.3-70b-versatile"),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tools_calls=True,
        search_knowledge=True,
        read_chat_history=True
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}")
    else:
        print(f"Continuing Run: {run_id}")
    
    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_assistant)



