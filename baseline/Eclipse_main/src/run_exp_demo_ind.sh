python run_exp_demo.py --rank 20  --num_seeds 5 --num_epochs 200 --command train --outdir ../results --todos_dir ../todos_train --best_config_file best_config.pkl --only_create_todos --datasets $1
python run_exp_demo.py --rank 20  --num_seeds 5 --num_epochs 200 --command train --outdir ../results --todos_dir ../todos_train --best_config_file best_config.pkl --datasets $1
python parser_ash_ind_utility.py --results_dir ../results
python run_exp_demo.py --rank 20  --num_seeds 5 --num_epochs 200 --command attack --outdir ../results --todos_dir ../todos_attack --best_config_file best_config.pkl --only_create_todos --datasets $1
python run_exp_demo.py --rank 20  --num_seeds 5 --num_epochs 200 --command attack --outdir ../results --todos_dir ../todos_attack --best_config_file best_config.pkl --datasets $1

python parser_ash_trans_attack.py --results_dir ./eval_twitch/RU
python parser_ash_trans_attack.py --results_dir ./eval_twitch/DE
python parser_ash_trans_attack.py --results_dir ./eval_twitch/FR
python parser_ash_trans_attack.py --results_dir ./eval_twitch/ENGB
python parser_ash_trans_attack.py --results_dir ./eval_twitch/PTBR