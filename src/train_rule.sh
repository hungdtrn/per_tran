bash run.sh -m src.tools.train_hoi override_cfg=src/config/hoi_per_tran.yaml model/architecture=hoi_per_tran_no_sqrt gpu=7 model/solver/scheduler=lambda_lr project_name=hoi_per_tran_no_sqrt experiment_name=rule wandb=true

bash run.sh -m src.tools.train_hoi override_cfg=src/config/hoi_per_tran.yaml model/architecture=hoi_per_tran_no_sqrt gpu=3 model/solver/scheduler=lambda_lr model.solver.optimizer.base_lr=0.001 project_name=hoi_per_tran_no_sqrt experiment_name=rule_lr0.001 wandb=true 


bash run.sh -m src.tools.train_hoi override_cfg=src/config/hoi_per_tran.yaml model/architecture=hoi_per_tran model.architecture.switch.switch_type=learn_distance_temp_gru model.architecture.switch.switch_feat=relative_location model.solver.loss.name=predict+switch dataset.future_window=5 gpu=6 model/solver/scheduler=lambda_lr project_name=hoi_per_tran experiment_name=learn_window5_switch_rel_loc wandb=true 

bash run.sh -m src.tools.train_hoi override_cfg=src/config/hoi_per_tran.yaml model/architecture=hoi_per_tran_no_sqrt model.architecture.switch.switch_type=learn_distance_temp_gru model.solver.loss.name=predict+switch dataset.future_window=5 model.solver.optimizer.base_lr=0.001 gpu=5 model/solver/scheduler=lambda_lr project_name=hoi_per_tran_no_sqrt experiment_name=learn_window5_lr0.001 wandb=true 

bash run.sh -m src.tools.train_hoi override_cfg=src/config/hoi_per_tran.yaml model/architecture=hoi_per_tran model.architecture.switch.switch_type=learn_distance_temp_gru model.architecture.switch.switch_feat=relative_location model.solver.loss.components=predict+switch dataset.future_window=4 gpu=4 model/solver/scheduler=lambda_lr project_name=hoi_per_tran_final experiment_name=learn_window4_switch_rel_loc wandb=true

bash run.sh -m src.tools.train_hoi override_cfg=src/config/hoi_crnn.yaml model/architecture=hoi_rnn gpu=6 project_name=hoi_rnn experiment_name=rnn wandb=true 

bash run.sh -m src.tools.train_hoi override_cfg=src/config/hoi_per_tran.yaml model/architecture=hoi_per_tran model.architecture.switch.switch_type=learn_distance_temp_gru model.architecture.switch.switch_feat=relative_location model.solver.loss.components=predict gpu=3 model/solver/scheduler=lambda_lr project_name=hoi_per_tran_final_bigger_obj experiment_name=learn_no_switch_loss wandb=true


bash run.sh -m src.tools.train_hoi override_cfg=src/config/hoi_per_tran.yaml model/architecture=hoi_per_tran model.architecture.switch.switch_type=learn_distance_temp_gru model.architecture.switch.switch_feat=relative_location model.architecture.name=per_tran_no_dist_detach model.solver.loss.components=predict gpu=3 model/solver/scheduler=lambda_lr project_name=hoi_per_tran_final_bigger_obj experiment_name=learn_no_switch_loss wandb=true