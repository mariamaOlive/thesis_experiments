from utils.prompt_builder import PromptBuilder
from semantic_kernel.agents.strategies import TerminationStrategy, KernelFunctionSelectionStrategy
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory, ChatHistoryTruncationReducer

from metric.rouge import ROUGE
from pydantic import Field

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



class AgentFunctions:
    
    def __init__(self, kernel):
        self.kernel = kernel
        # self.kernel = Kernel()
        # self.kernel.add_service(OpenAIChatCompletion(service_id="agents_interaction"))
    

    def get_selection_function(self, agent_1, agent_2, agent_3):
        
        file_path = "template/selection_function.yaml"
        prompt_template_selection = PromptBuilder.prompt_template(file_path)

        selection_function = KernelFunctionFromPrompt(
            function_name="selection",
            prompt_template_config=prompt_template_selection
            )
        

        history_reducer = ChatHistoryTruncationReducer(target_count=1)

        return KernelFunctionSelectionStrategy(
                    initial_agent=agent_1,
                    kernel = self.kernel,
                    function=selection_function,
                    result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else agent_1.name,
                    history_variable_name="lastmessage",
                    history_reducer=history_reducer,\
                    arguments=KernelArguments(name_agent_1=agent_1.name, name_agent_2=agent_2.name, name_agent_3=agent_3.name)
                )
        
    
    def get_termination_function(self, agent, ground_truth, max_iterations):
        
        history_reducer = ChatHistoryTruncationReducer(target_count=1)

        return ThresholdTerminationStrategy(
                summary_ground_truth = ground_truth,
                kernel = self.kernel,
                agents=[agent],
                history_variable_name = "lastmessage",
                maximum_iterations = max_iterations,
                history_reducer = history_reducer,
            )
    