import subprocess

subprocess.run(['pip','install','numpy==1.19.2'])
subprocess.run(['pip','install','matplotlib==3.3.2'])
subprocess.run(['pip','install','optuna==2.4.0'])

from . import simulation
from . import assessment_framework
from . import inference_framework
from . import simulation_colab
from . import assessment_framework_colab
from . import inference_framework_colab