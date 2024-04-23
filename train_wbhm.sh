#!/bin/bash

bash run.sh python -m src.wbhm.tools.train model/architecture=per_tran project_name=switch experiment_name=per_tran model.architecture.name=per_tran model.solver.scheduler.name=lambda_lr seed=10
