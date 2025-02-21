import json
from typing import Optional, Iterator
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Agentic_AI.logging import logger

from phi.agent import Agent
from phi.model.groq import Groq
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.utils.pprint import pprint_run_response

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class NewsArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")


class SearchResults(BaseModel):
    articles: list[NewsArticle]

class BlogPostGenerator(Workflow):

    searcher: Agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[DuckDuckGo()],
        instructions=["Given a topic, search for the top 5 articles and return the results in JSON format.",
                      "Ensure the output follows the SearchResults schema."],
        response_model=SearchResults,
        markdown=True
    )


    writer: Agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile"),
        instructions=[
            "You will be provided with a topic and a list of top articles on that topic.",
            "Carefully read each article and generate a New York Times worthy blog post on that topic.",
            "Break the blog post into sections and provide key takeaways at the end.",
            "Make sure the title is catchy and engaging.",
            "Always provide sources, do not make up information or sources.",
            "Return the blog post as a valid JSON object with keys: 'title', 'content', and 'source'."
        ],
        # structured_outputs=True,
        markdown=True
    )

    def run(self, topic: str, use_cache: bool = True) -> Iterator[RunResponse]:
        logger.info(f"Generating a blog post on: {topic}")

        if use_cache:
            cached_blog_post = self.get_cached_blog_post(topic)
            if cached_blog_post:
                yield RunResponse(content=cached_blog_post, event=RunEvent.workflow_completed)
                return
        
        search_results: Optional[SearchResults] = self.get_search_results(topic)

        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not find any articles on the topic: {topic}",
            )
            return

        yield from self.write_blog_post(topic, search_results)

    def get_cached_blog_post(self, topic: str) -> Optional[str]:

        logger.info("Checking if cached blog post exists.")
        return self.session_state.get("blog_post", {}).get(topic)
    
    def add_blog_post_to_cache(self, topic: str, blog_post: Optional[str]):

        logger.info(f"Saving blog post for topic: {topic}")
        self.session_state.setdefault("blog_posts", {})
        self.session_state["blog_posts"][topic] = blog_post
    
    def get_search_results(self, topic: str) -> Optional[SearchResults]:

        MAX_ATTEMPTS = 3

        for attempt in range(MAX_ATTEMPTS):
            try:
                searcher_response: RunResponse = self.searcher.run(topic)

                if not searcher_response or not searcher_response.content:

                    if not isinstance(searcher_response.content, SearchResults):
                        logger.warning(f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: Invalid response type")
                        continue

                    article_count = len(searcher_response.content.articles)
                    logger.info(f"Found {article_count} articles on attempt {attempt + 1}")
                    return searcher_response.content
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {str(e)}")

            logger.error(f"Failed to get search results after {MAX_ATTEMPTS} attempts")
            return None
    
    def write_blog_post(self, topic: str, search_results: SearchResults) -> Iterator[RunResponse]:

        logger.info("Writing blog post")

        writer_input = {"topic":topic, "articles": [v.model_dump() for v in search_results.articles]}

        yield from self.writer.run(writer_input, stream=True)

        self.add_blog_post_to_cache(topic, self.writer.run_response.content)


if __name__ == "__main__":
    from rich.prompt import Prompt

    topic = Prompt.ask(
        "[bold]Enter a blog post topic[/bold]\n",
        default="Current news in Delhi, India",
    )

    url_safe_topic = topic.lower().replace(" ", "-")

    generate_blog_post = BlogPostGenerator(
        session_id=f"generate-blog-post-one-{url_safe_topic}",
        storage=SqlWorkflowStorage(
            table_name="generate_blog_post_workflows",
            db_file="tmp/workflows.db",
        ),
    )

    blog_post: Iterator[RunResponse] = generate_blog_post.run(topic=topic, use_cache=True)

    pprint_run_response(blog_post, markdown=True)