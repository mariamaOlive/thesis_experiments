import os
import logging

from agent.extractor import ExtractorAgent
from agent.summarizer import SummarizerAgent
from agent.teacher import TeacherAgent
from agent.prompt_combine import PromptCombineAgent

from utils.rouge_plugin import RougePlugin
from utils.prompt_plugin import PromptPlugin
from utils.prompt_builder import PromptBuilder
from utils.agent_functions import AgentFunctions


from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

from prompt.prompt import (
    INITIAL_SUMMARIZER_PROMPT,
    EXTRACTOR_PROMPT
)

# logging.basicConfig(level=logging.INFO)

class ParallelOptimizer:
    def __init__(self, threshold: float = 0.7):
        
        self.EXTRACTOR_TEMPLATE_FILE = "template/extractor.yaml"
        self.SUMMARIZER_TEMPLATE_FILE = "template/summarizer.yaml"
        self.EVALUATOR_TEMPLATE_FILE = "template/evaluator.yaml"
        self.TEACHER_TEMPLATE_FILE = "template/teacher.yaml"
        self.COMBINE_TEMPLATE_FILE = "template/prompt_combine.yaml"
        self.debug_result = {}
        self.threshold = threshold
        
    def _create_kernel_with_chat_completion(self, service_id: str) -> Kernel:
        kernel = Kernel()
        kernel.add_service(OpenAIChatCompletion(service_id=service_id, api_key = os.getenv("OPENAI_API_KEY")))
        return kernel

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
            
            team = RoundRobinGroupChat([summarizer_agent, teacher_agent], termination_condition=termination)
            # Use `asyncio.run(...)` when running in a script.
            result = await team.run(task=INITIAL_SUMMARIZER_PROMPT)

            if result.stop_reason == "Text 'APPROVE' mentioned":
                data_prompt.append(result.messages[-2].content)

            print(f"Length data_prompt: {len(data_prompt)}")
            if len(data_prompt)==4:
                break
            
        # Create and run Prompt Combiner
        combine_agent =  PromptCombineAgent(COMBINE_NAME)
        extracted_text = await combine_agent.run_agent(data_prompt)
        print(f"Extracted text: {extracted_text}")
            
        # Show history of all best prompts
        with open("data_prompts.txt", "w", encoding="utf-8") as f:
            for line in data_prompt:
                f.write(line + "\n")
            
            