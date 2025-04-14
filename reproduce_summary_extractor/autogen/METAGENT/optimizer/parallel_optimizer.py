from agent.extractor import ExtractorAgent
from agent.summarizer import SummarizerAgent
from agent.teacher import TeacherAgent
from agent.prompt_combine import PromptCombineAgent


from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

from prompt.prompt import (
    INITIAL_SUMMARIZER_PROMPT,
    EXTRACTOR_PROMPT
)


class ParallelOptimizer:
    def __init__(self, threshold: float = 0.7):
        self.debug_result = {}
        self.threshold = threshold
        

    async def run(self, max_iterations: int, train_data: list[dict]):
        
        # Store different prompt from the interactions
        data_prompt = [] 
        EXTRACTOR_NAME = "Extractor"
        SUMMARIZER_NAME = "Summarizer"
        TEACHER_NAME = "Teacher"
        COMBINE_NAME = "Combiner"
        
        for i, data in enumerate(train_data):   
            
            # Ground truth value
            description = data["description"]
            # Readme value
            readme = data["readme"]
            
            print(f"Data #{i}:\n- Description: {description}")
            
            # Create and run Extractor Agent
            extractor_agent =  ExtractorAgent(EXTRACTOR_NAME)
            extracted_text = await extractor_agent.run_agent(EXTRACTOR_PROMPT, readme)
            print(f"Extracted text: {extracted_text}")
            
            
            # Chat conversation Teacher - Summarizer
            # Create Summarizer
            summarizer_agent = SummarizerAgent(
                name=SUMMARIZER_NAME,
                description='A agent that summarize READMEs based on the prompt provided by the Teacher agent',
                extracted_text=extracted_text, 
                ground_truth= description,
                threshold=self.threshold
                )

            # Create Teacher
            teacher_agent = TeacherAgent(
                name=TEACHER_NAME,
                description='A agent that improves the prompts utilized by the Summarizer agent',
                extracted_text=extracted_text,
                ground_truth=description)
            
            termination = MaxMessageTermination(max_iterations*2) |  TextMentionTermination("APPROVE")
            
            # Create Multi Agent team
            team = RoundRobinGroupChat([summarizer_agent, teacher_agent], termination_condition=termination)
            # Run team
            result = await team.run(task=INITIAL_SUMMARIZER_PROMPT)

            if result.stop_reason == "Text 'APPROVE' mentioned":
                data_prompt.append(result.messages[-2].content)

            print(f"Length data_prompt: {len(data_prompt)}")
            # if len(data_prompt)==4:
            #     break
            
        # Create and run Prompt Combiner
        combine_agent =  PromptCombineAgent(COMBINE_NAME)
        extracted_text = await combine_agent.run_agent(data_prompt)
        print(f"Extracted text: {extracted_text}")
            
        # Show history of all best prompts
        with open("data_prompts.txt", "w", encoding="utf-8") as f:
            for line in data_prompt:
                f.write(line + "\n")
            
            