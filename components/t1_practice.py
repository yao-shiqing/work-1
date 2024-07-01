from src.runners.episode_runner_01 import EpisodeRunner
from types import SimpleNamespace as SN
from utils.logging import Logger




runner = EpisodeRunner()
batch_value = runner.batch
print(batch_value)

