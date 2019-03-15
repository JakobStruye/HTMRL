# HTMRL
##Installing
`virtualenv -p python3 venv`  
`. venv/bin/active`  
`pip install -e .`  
All dependencies are set to a specific version known to work. Tested with Python3.7
##Running an experiment
The `.yml` files in `config` can be used  to configure your experiment.  
A main configuration for all subexperiments can be set, and then each subexperiment can override any setting. Use `enabled: 0` to avoid running some algorithms for some subexperiments.  
Run the experiment with `python run.py config/yourconfig.yml` To run only some subexperiments, add their names as additional parameters.  

Repeats of a single subexperiment will run in parallel, by default one per physical core.  

Per-step rewards are logged in output/, in a timestamped subdirectory. With several repeats, results of repeat `i` are found at lines `i*stepcount` to `(i+1)*stepcount`.
