# Rethinking the Influence of Source Code on Test Case Generation

To reproduce our experiments, we should first run:

```
python test_case_generation.py
python open_test_case_generation.py
```

to generate test cases for different prompts.

`Note: P_IC and P_CC use same solution of P_T_IC and P_T_CC while just remove the '""" """' part.`


Then, we run:

```
python divide_tests.py
python rq1_correcttestcase_correctness.py
python rq12_correctness.py
python rq13coverage.py
python rq14coverage.py
```

to process generated test cases for each RQ.

Finally, we can get the results in the paper by directly run:

```
python rqxx[a/b].py
```
to get each RQ's results.

For RQ4, we need to go to the pyter folder and then conduct experiments based on its environments.

To get RQ4 results, we run:

```python
cd pyter
bash run_pyter.sh
```
