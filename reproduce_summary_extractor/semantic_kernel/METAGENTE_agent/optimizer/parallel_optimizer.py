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
        self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT
        # self.extractor_agent = ExtractorAgent()
        # self.summarizer_agent = SummarizerAgent()
        self.teacher_agent = TeacherAgent()
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
            summarizer_agent = summarizer_agent_handler.create_agent(self.SUMMARIZER_TEMPLATE_FILE, extracted_text)
            summarizer_agent_handler.add_plugin_kernel("prompt_plugin", prompt_plugin)
            
            # Create Agent Evaluator 
            evaluator_agent_handler =  EvaluatorAgent(EVALUATOR_NAME)
            evaluator_agent = evaluator_agent_handler.create_agent(self.EVALUATOR_TEMPLATE_FILE, description)
            evaluator_agent_handler.add_plugin_kernel("prompt_plugin", prompt_plugin)
            evaluator_agent_handler.add_plugin_kernel("rouge_plugin", rouge_plugin)

            
            # Create Agent Teacher
            
            ########## BEGINNING: Agent Framework - Summarizer and Teacher ##########
            
            kernel = self._create_kernel_with_chat_completion("kernel_loop")
            
            initial_summarizer_prompt = """
            Summarize the following extracted text from a Github repository README into a short term/phrase introducing the repository:
            
            The output should include only a short term/phrase introducing the repository.
            """
            

            

            
            
            # EVALUATOR_INSTRUCTIONS = f"""
            # You are the Evaluator agent in a collaborative chat with a Teacher and a Summarizer.

            # Your role is to automatically evaluate the quality of the summary generated by the Summarizer using ROUGE-L, and update the system’s prompt storage **only if the prompt used by the Teacher led to a better summary** than all previous attempts.

            # You are given:
            # - The ground truth description of the GitHub repository:
            # <GROUND_TRUTH DESCRIPTION>
            # {description}
            # </GROUND_TRUTH DESCRIPTION>

            # Your tasks:
            # 1. Locate the most recent summary produced by the Summarizer.
            # 2. Locate the most recent prompt provided by the Teacher.
            # 3. Get the best rouge score stored.
            # 4. Calculate the ROUGE-L score between the generated summary and the ground truth.
            # 5. If this ROUGE-L score is higher than any previous score, update the stored best prompt with the most recent prompt of the Teacher
            # """
            # settings_evaluator = OpenAIChatPromptExecutionSettings(
            #                 ai_model_id="gpt-4o-mini",
            #                 temperature=0,
            #             )
            
            # agent_evaluator = ChatCompletionAgent(
            #         kernel=kernel,
            #         name=EVALUATOR_NAME,
            #         instructions=EVALUATOR_INSTRUCTIONS,
            #         # arguments=KernelArguments(settings = settings_evaluator)
            # )
            
            
            
            TEACHER_PROMPT = f"""
            You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short description term or phrase that captures the key concept or idea from the extracted text of a GitHub repository's README. Your job is to **improve the current summarization prompt** based on test results and past performance.

            # Inputs You Must Use:

            1. **Extracted README Text**  
            This is the raw input text that the LLM is summarizing.  
            <EXTRACTED_TEXT>
            {extracted_text}
            </EXTRACTED_TEXT>

            2. **Ground Truth Description**  
            This is the correct, ideal summary of the repository.  
            <GROUND_TRUTH_DESCRIPTION>
            {description}
            </GROUND_TRUTH_DESCRIPTION>

            3. **Last Generated Summary**  
            This is the most recent summary the LLM produced using the current prompt.  
            <GENERATED_SUMMARY>
            {{{{ prompt_plugin.GetLastSummary }}}}
            </GENERATED_SUMMARY>

            4. **Best ROUGE-L Score So Far**  
            This score measures the similarity between the generated summary and the ground truth.  
            <BEST_ROUGE_SCORE>
            {{{{ prompt_plugin.GetBestInstructionRougeScore }}}}
            </BEST_ROUGE_SCORE>

            5. **Current Best Instruction**  
            This is the prompt currently used to guide the LLM’s summarization.  
            <CURRENT_INSTRUCTION>
            {{{{ prompt_plugin.GetBestInstruction }}}}
            </CURRENT_INSTRUCTION>

            # Your Task:

            - **Step 1**: Read the extracted README text and the ground truth. If the ground truth appears verbatim (or nearly so) in the beginning of the README (e.g., tagline, overview), make sure your new instruction tells the LLM to prioritize that section.
            - **Step 2**: Review the last summary and ROUGE score. If the score is low or the summary is off-target, identify weaknesses or missing guidance in the current instruction.
            - **Step 3**: Improve the current instruction. Modify it with the smallest possible change that could improve performance. Do not rewrite the instruction from scratch.
            - **Step 4**: Output only the improved instruction. Do not include explanations, justifications, or any additional text.
            
            # Strict Rules:

            - Do NOT copy, reword, or hint at the ground truth description in the instruction.
            - Do NOT mention the ground truth in your output.
            - Do NOT include any explanation or labels in your output.
            - Your output must be only the improved instruction as a raw string.
            
            # Important:
            Store your response as the last instruction created by the Teacher.
            Make sure your output contains only the raw instruction — no extra text, explanations, or labels.
   
            """




            kernel_teacher = Kernel()
            kernel_teacher.add_service(OpenAIChatCompletion(service_id=TEACHER_NAME, ai_model_id="gpt-4o"))
            kernel_teacher.add_plugin(prompt_plugin, plugin_name= "prompt_plugin")
            kernel_teacher.add_plugin(rouge_plugin, plugin_name= "rouge_plugin")
            
            agent_teacher = ChatCompletionAgent(
                    kernel=kernel_teacher,
                    name=TEACHER_NAME,
                    instructions=TEACHER_PROMPT,
                    # arguments=KernelArguments(last_summary=prompt_plugin.get_last_summary(), 
                    #                           best_rouge_score=prompt_plugin.get_best_instruction_rouge_score(),
                    #                           best_instruction = prompt_plugin.get_best_instruction())
            )

    
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
                agents=[summarizer_agent,evaluator_agent,  agent_teacher],
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
            
            