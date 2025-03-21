import pandas as pd
from optimizer.parallel_optimizer import ParallelOptimizer
import asyncio


class Main:
    def __init__(self):
        self.num_iterations = 15
        self.threshold = .7
        # self.args = config_parser()

    def run(self):
        train_data = pd.read_csv("data/train_data.csv").to_dict(orient="records")

        optimizer = ParallelOptimizer(self.threshold)
        asyncio.run(optimizer.run(self.num_iterations, train_data))


if __name__ == "__main__":
    # load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
    main_flow = Main()
    main_flow.run()
