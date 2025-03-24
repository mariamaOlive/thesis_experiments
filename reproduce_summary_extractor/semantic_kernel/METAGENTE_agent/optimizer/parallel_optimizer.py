import os
import re

from metric.rouge import ROUGE
from prompt.prompt_agent import (
    COMBINE_PROMPT,
    EXTRACTOR_PROMPT,
    INITIAL_SUMMARIZER_PROMPT,
    # TEACHER_PROMPT,
)

from agent.extractor import ExtractorAgent
from agent.summarizer import SummarizerAgent
from agent.teacher import TeacherAgent
from agent.prompt_combine import PromptCombineAgent


from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import TerminationStrategy, KernelFunctionSelectionStrategy
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt
from semantic_kernel.contents import ChatHistory, ChatHistoryTruncationReducer
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig


from pydantic import Field

class ThresholdTerminationStrategy(TerminationStrategy):
    summary_ground_truth: str = Field(...)
    threshold: float = 0.7

    def __init__(self, summary_ground_truth: str, **kwargs):
        # this ensures pydantic sees summary_ground_truth
        super().__init__(summary_ground_truth=summary_ground_truth, **kwargs)

    async def should_agent_terminate(self, agent, history):
        try:
            match = re.search(r"<GENERATED_DESCRIPTION>(.*?)</GENERATED_DESCRIPTION>", history[-1].content, re.DOTALL)
            if match:
                short_description = match.group(1).strip()
                print(short_description)
            rougeL_score = ROUGE().get_RougeL(
                string_1=match,
                string_2=self.summary_ground_truth
            )
            return rougeL_score >= self.threshold
        except Exception:
            return False



class ParallelOptimizer:
    def __init__(self, threshold: float = 0.7):
        self.extractor_prompt = EXTRACTOR_PROMPT
        self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT
        self.extractor_agent = ExtractorAgent()
        self.summarizer_agent = SummarizerAgent()
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
        data_prompt = [] ## CHECK: The way that is now stores the one that reached the rouge threshold
        data_prompt.append(self.summarizer_prompt) ## CHECK: I added the initial prompt 
        
        for i, data in enumerate(train_data):   
            
            # Load the initial prompt created for the experiment
            self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT  ## CHECK: Should I put this outside because it is reinitializing every iteration, so should each iteraction not remember previous loop 
        
            # Ground truth value
            description = data["description"]
            # Readme value
            readme = data["readme"]
            
            print(f"Data #{i}:\n- Description: {description}")
            
            best_score = 0
            best_summarizer_prompt = self.summarizer_prompt  # Why reload always with the same initial prompt? #Is it just to initialize?
            
            # Run Extractor Agent
            # extracted_text = await self.extractor_agent.run(
            #     prompt=self.extractor_prompt, readme_text=readme
            # )
            
            ########## BEGINNING: Agent Framework - Extractor ##########
            # Adding settings of the agent
            settings = OpenAIChatPromptExecutionSettings(
                            service_id="extractor",
                            ai_model_id="gpt-4o-mini",
                            temperature=0,
                        )
            # Create the agent
            agent_extractor = ChatCompletionAgent(
                    kernel = self._create_kernel_with_chat_completion("extractor"),
                    name="extractor", 
                    instructions="self.extractor_prompt",
                    arguments=KernelArguments(settings=settings)
                    )
            chat = ChatHistory()
            chat.add_user_message(readme)
            # Generate the agent response
            extracted_text = await agent_extractor.get_response(chat)
            ########## END: Agent Framework ########## 
            extracted_text = str(extracted_text)
            print(extracted_text)
            
            
            ########## BEGINNING: Agent Framework - Summarizer and Teacher ##########
            SUMMARIZER_NAME = "Summarizer"
            TEACHER_NAME = "Teacher"
            kernel = self._create_kernel_with_chat_completion("kernel_loop")
            
            # settings_summarizer = OpenAIChatPromptExecutionSettings(
            #                 ai_model_id="gpt-4o-mini",
            #                 temperature=0,
            #             )
            
            SUMMARIZER_PROMPT = """
            You are a Summarizer agent in a multi-agent system.

            Your task is to generate a short term or phrase summarizing the core idea or purpose of a GitHub repository, using the README text provided.

            Instructions:
            - You will receive the README content inside <README>...</README> tags.
            - You will also receive a prompt to follow inside <PROMPT>...</PROMPT> tags.
            - Apply the prompt to the README content.
            - Return both:
                - The prompt you used (inside <PROMPT>)
                - The generated description (inside <GENERATED_DESCRIPTION>)

            Output format:
            <PROMPT>...used prompt...</PROMPT>
            <GENERATED_DESCRIPTION>...your one-line summary...</GENERATED_DESCRIPTION>
            """


            
            agent_summarizer = ChatCompletionAgent(
                    kernel=kernel,
                    name=SUMMARIZER_NAME,
                    instructions=SUMMARIZER_PROMPT,
                    # arguments=KernelArguments(settings=settings_summarizer)
            )
            
            
            # settings_teacher = OpenAIChatPromptExecutionSettings(
            #                 ai_model_id="gpt-4o-mini",
            #                 temperature=0.7,
            #             )
            TEACHER_PROMPT =  """
            You are a professional Prompt Engineer in a multi-agent system.

            Your job is to iteratively improve the Summarizer's prompt to produce better one-line descriptions of GitHub repositories based on their README.

            You are given in the chat:
            - The most recent Summarizer output, which contains:
                - <PROMPT>...</PROMPT>: the prompt it used
                - <GENERATED_DESCRIPTION>...</GENERATED_DESCRIPTION>: the description it generated
            - The latest <README>...</README> content (always provided in recent user/system messages)
            - The <GROUND_TRUTH DESCRIPTION>...</GROUND_TRUTH DESCRIPTION> (used for evaluation only)

            Your task:
            1. Compare the generated description to the ground truth.
            2. If the ground truth appears early in the README (as a tagline or overview), include in the prompt that such parts should be prioritized.
            3. Identify what’s missing or misleading in the current prompt and make **minimal changes**.
            4. DO NOT mention or refer to the ground truth in the new prompt.
            5. DO NOT provide explanations, reasoning, or labels — only return a clean prompt.

            Output format:
            <PROMPT>...the improved prompt only...</PROMPT>
            """

    
            agent_teacher = ChatCompletionAgent(
                    kernel=kernel,
                    name=TEACHER_NAME,
                    instructions=TEACHER_PROMPT,
                    # arguments=KernelArguments(settings=settings_teacher)
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

                Always follow these rules when selecting the next participant:
                - If RESPONSE is user input, it is {SUMMARIZER_NAME}'s turn.
                - If RESPONSE is by {SUMMARIZER_NAME}, it is {TEACHER_NAME}'s turn.
                - If RESPONSE is by {TEACHER_NAME}, it is {SUMMARIZER_NAME}'s turn.

                RESPONSE:
                {{{{$lastmessage}}}}
                """
            )
            
            # history_reducer = ChatHistoryTruncationReducer(target_count=1)
            
            group_chat = AgentGroupChat(
                agents=[agent_summarizer, agent_teacher],
                selection_strategy=KernelFunctionSelectionStrategy(
                    initial_agent=agent_summarizer,
                    kernel = kernel,
                    function=selection_function,
                    result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else SUMMARIZER_NAME,
                    history_variable_name="lastmessage",
                    # history_reducer=history_reducer,
                ),
                termination_strategy=ThresholdTerminationStrategy(
                    summary_ground_truth = description,
                    kernel = kernel,
                    agents=[agent_summarizer],
                    history_variable_name="lastmessage",
                    maximum_iterations=15,
                    # history_reducer=history_reducer,
                ),
            )
            
            initial_prompt =f""" 
            <README>{description}</README>

            <PROMPT>
            Summarize the extracted text from a Github repository README into a short term/phrase introducing the repository.
            </PROMPT>
            """
            await group_chat.add_chat_message(message=initial_prompt)
            async for content in group_chat.invoke():
                print(f"# {content.name}: {content.content}")
         
            print("#############################################\n\n")

            if i==1:
                break
            
            print(f"Length data_prompt: {len(data_prompt)}")
            
            
        # Show history of all best prompts
        with open("data_prompts.txt", "w", encoding="utf-8") as f:
            for line in data_prompt:
                f.write(line + "\n")
            
            