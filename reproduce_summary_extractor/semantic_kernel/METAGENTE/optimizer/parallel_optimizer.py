from metric.rouge import ROUGE
from prompt.prompt import (
    COMBINE_PROMPT,
    EXTRACTOR_PROMPT,
    INITIAL_SUMMARIZER_PROMPT,
    TEACHER_PROMPT,
)

from agent.extractor import ExtractorAgent
from agent.summarizer import SummarizerAgent
from agent.teacher import TeacherAgent
from agent.prompt_combine import PromptCombineAgent


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
            extracted_text = await self.extractor_agent.run(
                prompt=self.extractor_prompt, readme_text=readme
            )
            
            print(extracted_text)
            
            for iter in range(max_iterations):
                print(f"\nIteration #{iter}:")

                print(f"Extracted Text: {extracted_text}")
                
                about = await self.summarizer_agent.run(
                    prompt=self.summarizer_prompt, extracted_text=extracted_text
                )
                
                print(f"\nGenerated About: {about}")

                rougeL_score = ROUGE().get_RougeL(string_1=about, string_2=description)
                rouge1_score = ROUGE().get_Rouge1(string_1=about, string_2=description)
                rouge2_score = ROUGE().get_Rouge2(string_1=about, string_2=description)

                print(f"\nRouge1 Score: {rouge1_score}")
                print(f"Rouge2 Score: {rouge2_score}")
                print(f"RougeL Score: {rougeL_score}")
                
                # Replacing the best score found
                if rougeL_score > best_score:
                    best_score = rougeL_score
                    best_summarizer_prompt = self.summarizer_prompt #Getting the best prompt so far 
                    
                # Verify if is below the defined threshold
                # If it is not achieved send the prompts and metric to the teacher agent
                if rougeL_score < self.threshold:
                    self.summarizer_prompt = await self.teacher_agent.run(
                        extracted_text=extracted_text,
                        description=description,
                        generated_about=about,
                        rouge_score=rougeL_score,
                        summarizer_prompt=self.summarizer_prompt,
                    )
                    print(f"\nNew Summarizer Prompt: {self.summarizer_prompt}")
                    
                # If the threshold is achieved just store the summarizer prompt without
                # going to the teacher and stop the loop
                else:
                    data_prompt.append(best_summarizer_prompt)
                    break
                
                print(f"Best RougeL Score for Data #{i}: {best_score}")
                
            self.summarizer_prompt = await self.prompt_combine.run(prompt_list=data_prompt)
            print(f"Final Result:\nSummarizer Prompt: {self.summarizer_prompt}")

            print("#############################################\n\n")

            if i==0:
                break
            
            print(f"Length data_prompt: {len(data_prompt)}")
            
            
        # Show history of all best prompts
        with open("data_prompts.txt", "w", encoding="utf-8") as f:
            for line in data_prompt:
                f.write(line + "\n")
            
            