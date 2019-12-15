# The bouncer problem (code for experiments)

## Code for the experiment proposed in Section 4.2 of the paper "The Bouncer Problem: Challenges to Remote Explainability".
Code author: Erwan Le Merrer (elemerre@acm.org)

The code leverages the German Credit Dataset, available at https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data).

Rationale: the code is composed of 3 parts:
* Train a model to predict credit default based on all features, including discriminative ones (i.e., age, sex, employment,foreigner)
* Train a model on all but the discriminative features
* This part simply computes the fraction of label changes (ie, IPs) between the prediction of the original and the model without discriminative features
   (the four discriminatory features of each of 50 test profiles are sequentially replaced by the ones of the 49 remaining profiles; each resulting test profile is fed to the model for prediction)
   The results are used as a basis for Figure 4, after averaging 30 trials.
   
## Requirements and execution
 
 Requires: Python, Keras, Tensorflow, numpy and Pandas
 Execution: 
 
```python
 python bouncer_problem_neural.py
```
Example run results:
```
*** Percentage of incoherent pairs (IPs) ***
All four: 0.026122
Employment:  0.008163
Sex/status: 0.002449
Age: 0.008571
Foreigner: 0.004898
```

(run with the following library versions: Python 3.6, Keras 2.2.4, Tensorflow 1.14.0)
