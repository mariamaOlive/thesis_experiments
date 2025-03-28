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
        # data_prompt.append(self.summarizer_prompt) ## CHECK: I added the initial prompt 
        
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
            EXTRACTOR_PROMPT = """
            Your task is to shorten and extract only the introduction and description information from the README of a Github repository. You are given a README text from a GitHub repository:

            # Steps
            - **Identify the structure of the repository**: The README file is a structure text file that might contains many sections such as introduction, description, installation, contributing, license,...
            - **Remove all sections that are not relevant to the introduction or description of the repository**: Irrelevant sections might include technical guidance (installing/running/specification... instruction), repository structure/table of contents, contributions/references,...
            - **Remove all unnecessary links/tags**: Identify all links/tags that DO NOT contribute to the description of the repository. You must remove all of these reference links and tags.
            - **Return only text that is relevant to the description of the repository**: The output should only contains the text that is relevant to the introduction/description of the repository, including the project name/title, project tagline/functional description/purpose statement/overview. DO NOT include any output identifications such as: "Here's the ..." or "Extracted README:"
            """
            
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
                    instructions=EXTRACTOR_PROMPT,
                    arguments=KernelArguments(settings=settings)
                    )
            chat = ChatHistory()
            chat.add_user_message(readme)
            # Generate the agent response
            extracted_text = await agent_extractor.get_response(chat)
            extracted_text = extracted_text.content
            print(f"Extracted text: {extracted_text}")
            ########## END: Agent Framework ########## 
            
            
            ########## BEGINNING: Agent Framework - Summarizer and Teacher ##########
            SUMMARIZER_NAME = "Summarizer"
            TEACHER_NAME = "Teacher"
            EVALUATOR_NAME = "Evaluator"
            kernel = self._create_kernel_with_chat_completion("kernel_loop")
            rouge = RougePlugin()
            
            initial_summarizer_prompt = """
            Summarize the following extracted text from a Github repository README into a short term/phrase introducing the repository:
            
            The output should include only a short term/phrase introducing the repository.
            """
            prompt_plugin = PromptPlugin(initial_summarizer_prompt)
            kernel.add_plugin(rouge, plugin_name= "rouge_plugin")
            kernel.add_plugin(prompt_plugin, plugin_name= "prompt_plugin")
            
            kernel_summarizer = Kernel()
            kernel_summarizer.add_service(OpenAIChatCompletion(service_id=SUMMARIZER_NAME, ai_model_id="gpt-4o-mini"))
            # settings_summarizer = OpenAIChatPromptExecutionSettings(
            #                 service_id=SUMMARIZER_NAME, 
            #                 ai_model_id="gpt-4o-mini",
            #                 temperature=0,
            #             )
            
            
            SUMMARIZER_INSTRUCTIONS = f"""
            You are the Summarizer agent.

            You are given:
            - The extracted README text: 
            <README>
            {extracted_text}
            </README>

            Your Task:
            1. Locate the best instructions created by the Teacher.
            2. Use those instructions and apply them to the README.
            3. Set the last summary description based on the output of step 2.
            4. Output only the generated summary description.

            Do not include explanations, introductions, or meta-comments.
            Do not repeat or reference the instructions.
            Output only the concise summary description as plain text.
            """



            agent_summarizer = ChatCompletionAgent(
                    kernel=kernel,
                    name=SUMMARIZER_NAME,
                    instructions=SUMMARIZER_INSTRUCTIONS,
                    # arguments=KernelArguments(settings = settings_summarizer)
            )
            
            
            EVALUATOR_INSTRUCTIONS = f"""
            You are the Evaluator agent in a collaborative chat with a Teacher and a Summarizer.

            Your role is to automatically evaluate the quality of the summary generated by the Summarizer using ROUGE-L, and update the system’s prompt storage **only if the prompt used by the Teacher led to a better summary** than all previous attempts.

            You are given:
            - The ground truth description of the GitHub repository:
            <GROUND_TRUTH DESCRIPTION>
            {description}
            </GROUND_TRUTH DESCRIPTION>

            Your tasks:
            1. Locate the most recent summary produced by the Summarizer.
            2. Locate the most recent prompt provided by the Teacher.
            3. Get the best rouge score stored.
            4. Calculate the ROUGE-L score between the generated summary and the ground truth.
            5. If this ROUGE-L score is higher than any previous score, update the stored best prompt with the most recent prompt of the Teacher
            """
            settings_evaluator = OpenAIChatPromptExecutionSettings(
                            ai_model_id="gpt-4o-mini",
                            temperature=0,
                        )
            
            agent_evaluator = ChatCompletionAgent(
                    kernel=kernel,
                    name=EVALUATOR_NAME,
                    instructions=EVALUATOR_INSTRUCTIONS,
                    arguments=KernelArguments(settings = settings_evaluator)
            )
            
            
            
            TEACHER_PROMPT = f"""
            You are the Teacher agent in a chat focused on building the best possible **summarization prompt** for a Summarizer agent.

            You are **not** generating summaries yourself.

            Your only task is to write the instructions that the Summarizer will follow in their next message. Your output will **become the new summarization instructions**.

            Goal:
            Refine the summarization prompt so that the Summarizer can extract a **short term/phrase** that best captures the purpose or functionality of a GitHub repository from its README.

            You are given:
            - Extracted README text:
            <README>
            {extracted_text}
            </README>

            - Ground truth description:
            <GROUND_TRUTH DESCRIPTION>
            {description}
            </GROUND_TRUTH DESCRIPTION>

            Your Task:
            1. Locate the best summarization instruction created so far by the Teacher. 
            2. Observe the Summarizer and Evaluator outputs to guide changes in the new instruction. (Previous messages in the chat)
            3. Improve the last best instruction.
            - **You must use the best instruction as the base**: This is an optimization task, so each new instruction must be a minimally modified version of the current best instruction.
            - **Prioritize extracting an existing tagline, functional description, purpose statement, or overview near the beginning of the README**: If the ground truth aligns with a clear phrase or description at the beginning of the README, the instruction must emphasize prioritizing that part of the text.
            - **Preserve and minimally revise**: Make only the smallest necessary changes to the current best instruction to fix issues or improve specificity. Do not completely rewrite.
            - **You must make a change** to avoid stagnation and continue improving.
            - Use the ground truth **only** as a reference to evaluate performance — never incorporate, hint at, or paraphrase it.

            Strictly DO NOT:
            - Mention the ground truth in your output.
            - Copy, paraphrase, or include any part of the ground truth in the new instruction.
            - Reword or “hint at” the ground truth in the instruction.
            - Provide any explanation, justification, or labels (e.g., “Prompt:”, “New Prompt:”, etc.).

            Output:
            - Only output the improved instruction string — nothing else.
            - Do not update the best instruction — that is the Evaluator’s responsibility.
            """


            kernel_teacher = Kernel()
            kernel_teacher.add_service(OpenAIChatCompletion(service_id=SUMMARIZER_NAME, ai_model_id="gpt-4o"))
            agent_teacher = ChatCompletionAgent(
                    kernel=kernel,
                    name=TEACHER_NAME,
                    instructions=TEACHER_PROMPT,
                    # arguments=KernelArguments(settings = settings_teacher)
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
            
            history_reducer = ChatHistoryTruncationReducer(target_count=2)
            
            group_chat = AgentGroupChat(
                agents=[agent_summarizer,agent_evaluator,  agent_teacher],
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
                    agents=[agent_evaluator],
                    history_variable_name="lastmessage",
                    maximum_iterations=30,
                    # history_reducer=history_reducer,
                ),
            )
            
            await group_chat.add_chat_message(message="start the tasks")
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
            
            