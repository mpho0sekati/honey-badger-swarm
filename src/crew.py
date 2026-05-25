import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

def get_groq_llm():
    """Configures Groq as the LLM provider."""
    return LLM(
        model="groq/llama-3.3-70b-versatile", # Or 'llama-3.1-8b-instant'
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

def run_crew(topic: str):
    llm = get_groq_llm()
    search_tool = SerperDevTool()

    # Pass the llm instance to the agent
    researcher = Agent(
        role='Researcher',
        goal=f'Research the topic: {topic}',
        backstory='Expert researcher.',
        tools=[search_tool],
        llm=llm
    )

    task = Task(description=f'Research {topic} and summarize.', expected_output='A summary.')

    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)
    return crew.kickoff(inputs={'topic': topic})
