python run_exp_demo.py --rank 20  --num_seeds 5 --command train --outdir ../results --todos_dir ../todos_train --best_config_file best_config.pkl --only_create_todos --datasets $1
python run_exp_demo.py --rank 20  --num_seeds 5 --command train --outdir ../results --todos_dir ../todos_train --best_config_file best_config.pkl --datasets $1
python parser_ash.py --results_dir ../results
python run_exp_demo.py --rank 20  --num_seeds 5 --command attack --outdir ../results --todos_dir ../todos_attack --best_config_file best_config.pkl --only_create_todos --datasets $1
python run_exp_demo.py --rank 20  --num_seeds 5 --command attack --outdir ../results --todos_dir ../todos_attack --best_config_file best_config.pkl --datasets $1

array=()
IFS=',' 
read -r -a array <<< "$1"
for dataset in "${array[@]}";
do  
    if [ "$dataset" == "Cora" ]
    then 
        python parser_ash_trans_attack.py --results_dir ./eval_cora
    elif [ "$dataset" == "CiteSeer" ]
    then
        echo "$dataset"
        python parser_ash_trans_attack.py --results_dir ./eval_citeseer
    elif [ "$dataset" == "PubMed" ]
    then
        python parser_ash_trans_attack.py --results_dir ./eval_pubmed
    elif [ "$dataset" == "facebook_page" ] 
    then
        python parser_ash_trans_attack.py --results_dir ./eval_facebook_page
    elif [ "$dataset" == "Chameleon" ]
    then
        python parser_ash_trans_attack.py --results_dir ./eval_chameleon
    elif [ "$dataset" == "TwitchES" ]
    then
        python parser_ash_trans_attack.py --results_dir ./eval_twitch/ES
    elif [ "$dataset" == "TwitchRU" ]
    then
        python parser_ash_trans_attack.py --results_dir ./eval_twitch/RU
    elif [ "$dataset" == "TwitchDE" ]
    then
        python parser_ash_trans_attack.py --results_dir ./eval_twitch/DE
    elif [ "$dataset" == "TwitchFR" ]
    then
        python parser_ash_trans_attack.py --results_dir ./eval_twitch/FR
    elif [ "$dataset" == "TwitchENGB" ]
    then
        python parser_ash_trans_attack.py --results_dir ./eval_twitch/ENGB
    elif [ "$dataset" == "TwitchPTBR" ]
    then
        python parser_ash_trans_attack.py --results_dir ./eval_twitch/PTBR
    fi
done