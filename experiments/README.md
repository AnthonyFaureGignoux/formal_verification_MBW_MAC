# Theoretical analysis

We compare our strategy to OQA strategy.

## Setup

*RQ:* 
The symbol `$` denotes terminal commands.

1. Install the required package:
`$ !pip install -r requirements.txt`

2. Run *main_file.py*:
`$ python main_file.py`
If you want to record the execution within a file, you may prefer the command:
`$ python main_file.py > result.txt`

3. Interact with the program to set the parameters
    * Model selection: either Resnet-18 or VGG-16
    * Model detail: whether print the pytorch module
    * Data dictionary: whether print the summary table (a detailed reinterpretation of pytorch module). If yes, you have to choose which architecture you want.

## Files

There are three python files:

* `main_file.py`:
It is one that calls the others files, interacts with the users and define the MAC profile. The MAC profile given is the one for *Full-Precision* computation, *i.e.*, FP(a,b).
* `mac_profile.py`:
The class that enables the comparison. It defines how many operations are required to perform a MAC. Note that we call *MAC* the required computation, *i.e.*, FP(a,b).
* `functions.py`: 
It gathers all the functions needed to achieve the comparison.