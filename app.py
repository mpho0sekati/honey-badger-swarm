from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

def run_crew(topic: str):
    # Define Tools
    search_tool = SerperDevTool()

    # Define Agents
    researcher = Agent(
        role='Researcher',
        goal=f'Research the topic: {topic}',
        backstory='Expert researcher.',
        tools=[search_tool]
    )

    # Define Tasks
    task = Task(description=f'Research {topic} and summarize.', expected_output='A summary.')

    # Define Crew
    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)
    return crew.kickoff(inputs={'topic': topic})
