# How to run EMG removal experiments
First install the requirements needed:
```shell
pip install -r requirements.txt
```

Next, run the script:
```shell
python3 emg_removal_experiment.py --method=methodname --threshold=threshold
```
- `methodname` can be either `cca`, `uica` or `fastica`. `fastica` is currently not working properly so please go with `cca` or `uica`
- `threshold`: If the predicted probability of a component is greater than `threshold` then the model will choose that component. The greater the threshold, the more conservative the model will be in picking EMG (according to the model itself, not according to the expert). can be any floating point number between 0 and 1. The default value is 0.5
