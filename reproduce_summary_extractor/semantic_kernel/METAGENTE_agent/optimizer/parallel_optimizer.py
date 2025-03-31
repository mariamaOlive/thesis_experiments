import os
import re
import logging

from metric.rouge import ROUGE
from prompt.prompt import (
    COMBINE_PROMPT,
    EXTRACTOR_PROMPT,
    INITIAL_SUMMARIZER_PROMPT,
    # TEACHER_PROMPT,
)

from agent.extractor import ExtractorAgent
from agent.summarizer import SummarizerAgent
from agent.evaluator import EvaluatorAgent
from agent.teacher import TeacherAgent
from agent.prompt_combine import PromptCombineAgent

from utils.rouge_plugin import RougePlugin
from utils.prompt_plugin import PromptPlugin
from utils.prompt_builder import PromptBuilder


from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import TerminationStrategy, KernelFunctionSelectionStrategy
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt
from semantic_kernel.contents import ChatHistory, ChatHistoryTruncationReducer
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig


from pydantic import Field
# logging.basicConfig(level=logging.INFO)

class ThresholdTerminationStrategy(TerminationStrategy):
    summary_ground_truth: str = Field(...)
    threshold: float = 0.7

    def __init__(self, summary_ground_truth: str, **kwargs):
        # this ensures pydantic sees summary_ground_truth
        super().__init__(summary_ground_truth=summary_ground_truth, **kwargs)

    async def should_agent_terminate(self, agent, history):
        try:
            if agent.name == "Evaluator":
                
                short_description = next((msg.content for msg in reversed(history) if msg.name == "Summarizer"), None)

                rougeL_score = ROUGE().get_RougeL(
                    string_1=short_description,
                    string_2=self.summary_ground_truth
                )
                return rougeL_score >= self.threshold
            else: 
                return False
        except Exception:
            return False


class ParallelOptimizer:
    def __init__(self, threshold: float = 0.7):
        
        self.EXTRACTOR_TEMPLATE_FILE = "template/extractor.yaml"
        self.SUMMARIZER_TEMPLATE_FILE = "template/summarizer.yaml"
        self.EVALUATOR_TEMPLATE_FILE = "template/evaluator.yaml"
        self.TEACHER_TEMPLATE_FILE = "template/teacher.yaml"
        self.COMBINE_TEMPLATE_FILE = "template/combine.yaml"
        self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT
        # self.extractor_agent = ExtractorAgent()
        # self.summarizer_agent = SummarizerAgent()
        # self.teacher_agent = TeacherAgent()
        self.prompt_combine = PromptCombineAgent()
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
        EVALUATOR_NAME = "Evaluator"
        
        for i, data in enumerate(train_data):   
            
            # Ground truth value
            description = data["description"]
            # Readme value
            readme = data["readme"]
            
            print(f"Data #{i}:\n- Description: {description}")
            
            # Create Extractor Agent
            extractor_agent_handler =  ExtractorAgent(EXTRACTOR_NAME)
            extractor_agent = extractor_agent_handler.create_agent(self.EXTRACTOR_TEMPLATE_FILE)
            
            # Create Chat Extractor
            chat = ChatHistory()
            chat.add_user_message(readme)
        
            # Start Conversation Chat Extractor
            extracted_text = await extractor_agent.get_response(chat)
            extracted_text = extracted_text.content
            print(f"Extracted text: {extracted_text}")
            
            # Initialize plugins
            rouge_plugin = RougePlugin()
            prompt_plugin = PromptPlugin()
            
            # Create Agent Summarizer
            summarizer_agent_handler =  SummarizerAgent(SUMMARIZER_NAME)
            summarizer_agent_handler.add_plugin_kernel("prompt_plugin", prompt_plugin)
            summarizer_agent = summarizer_agent_handler.create_agent(self.SUMMARIZER_TEMPLATE_FILE, extracted_text)
            
            # Create Agent Evaluator 
            evaluator_agent_handler =  EvaluatorAgent(EVALUATOR_NAME)
            evaluator_agent_handler.add_plugin_kernel("prompt_plugin", prompt_plugin)
            evaluator_agent_handler.add_plugin_kernel("rouge_plugin", rouge_plugin)
            evaluator_agent = evaluator_agent_handler.create_agent(self.EVALUATOR_TEMPLATE_FILE, description)

            
            # Create Agent Teacher
            teacher_agent_handler =  TeacherAgent(TEACHER_NAME)
            teacher_agent_handler.add_plugin_kernel("prompt_plugin", prompt_plugin)
            teacher_agent_handler.add_plugin_kernel("rouge_plugin", rouge_plugin)
            teacher_agent = teacher_agent_handler.create_agent(self.TEACHER_TEMPLATE_FILE, ground_truth=description, extracted_text=extracted_text)

            
            
            ########## BEGINNING: Agent Framework - Summarizer and Teacher ##########
            
            kernel = self._create_kernel_with_chat_completion("kernel_loop")
            
            initial_summarizer_prompt = """
            Summarize the following extracted text from a Github repository README into a short term/phrase introducing the repository:
            
            The output should include only a short term/phrase introducing the repository.
            """
    
            selection_function = KernelFunctionFromPrompt(
                function_name="selection",
                prompt=f"""
            Determine which participant takes the next turn in a conversation based on the most recent participant.
            State only the name of the participant to take the next turn.
            No participant should take more than one turn in a row.

            Choose only from these participants:
            - {SUMMARIZER_NAME}
            - {TEACHER_NAME}
            - {EVALUATOR_NAME}

            Rules:
            - If RESPONSE is user input, it is {SUMMARIZER_NAME}'s turn.
            - If RESPONSE is by {TEACHER_NAME}, it is {SUMMARIZER_NAME}'s turn.
            - If RESPONSE is by {SUMMARIZER_NAME}, it is {EVALUATOR_NAME}'s turn.
            - If RESPONSE is by {EVALUATOR_NAME}, it is {TEACHER_NAME}'s turn.

            RESPONSE:
            {{{{ $lastmessage }}}}
            """
            )
            
            history_reducer = ChatHistoryTruncationReducer(target_count=1)
            
            group_chat = AgentGroupChat(
                agents=[summarizer_agent,evaluator_agent,  teacher_agent],
                selection_strategy=KernelFunctionSelectionStrategy(
                    initial_agent=summarizer_agent,
                    kernel = kernel,
                    function=selection_function,
                    result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else SUMMARIZER_NAME,
                    history_variable_name="lastmessage",
                    history_reducer=history_reducer,
                ),
                termination_strategy=ThresholdTerminationStrategy(
                    summary_ground_truth = description,
                    kernel = kernel,
                    agents=[evaluator_agent],
                    history_variable_name="lastmessage",
                    maximum_iterations=30,
                    history_reducer=history_reducer,
                ),
            )
            
            await group_chat.add_chat_message(message=initial_summarizer_prompt)
            async for content in group_chat.invoke():
                print(f"# {content.name}: {content.content}")
                
            if(prompt_plugin.best_rouge_score>.7):
                data_prompt.append(prompt_plugin.best_instruction)
    
            print("#############################################\n\n")

            print(f"Length data_prompt: {len(data_prompt)}")
            if len(data_prompt)==4:
                break
            
        # Call Prompt Combiner
        summarizer_list = PromptBuilder._clean_prompt_list(data_prompt)
        
        COMBINER_PROMPT = f"""
        You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted text of the README of a Github repository. Your task is to combine several candidate prompts for the LLM into a final prompt.
        
        # Steps:
        - **Review all candidate prompts**: Analyze the following prompts to identify common parts to be included in the final prompt and also includes specific details or conditional key points from these prompts to be included in the final prompt
        <CANDIDATE_PROMPTS>
        {summarizer_list}
        </CANDIDATE_PROMPTS>
        - **Generate a final prompt**: Based on the common parts and conditional key points, generate a final prompt for the LLM.

        # Output Format:
        Do not include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the prompt for the LLM
        """
        
        print(COMBINER_PROMPT)
        
        # Adding settings of the agent
        settings = OpenAIChatPromptExecutionSettings(
                        service_id="Combiner",
                        ai_model_id="gpt-4o",
                        temperature=.7,
                    )
        
        kernel_combiner = Kernel()
        kernel_combiner.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o", service_id="Combiner"))
        # Create the agent
        agent_combiner = ChatCompletionAgent(
                kernel = kernel_combiner,
                name="Combiner", 
                instructions=COMBINER_PROMPT,
                arguments=KernelArguments(settings=settings)
                )
        chat_combiner = ChatHistory()
        chat_combiner.add_user_message("Start task")
        # Generate the agent response
        extracted_text = await agent_combiner.get_response(chat)
        extracted_text = extracted_text.content
        print(f"Extracted text: {extracted_text}")
            
        # Show history of all best prompts
        with open("data_prompts.txt", "w", encoding="utf-8") as f:
            for line in data_prompt:
                f.write(line + "\n")
            
            