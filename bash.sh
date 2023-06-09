# P1 Training
python scripts/train.py --dataset msvd --use_z --kl_rate 0.2 --learning_rate 0.015 --pathToData datasets/subsets/verb/msvd/msvd_all_centerK6_verb.json --mark P1K6

python scripts/train.py --dataset msrvtt --use_z --kl_rate 0.5 --learning_rate 0.015 --pathToData datasets/subsets/verb/msrvtt/msrvtt_all_center_K5.json --mark P1K5

python scripts/train.py --dataset vatex --use_z --kl_rate 0.9 --mark K3 --pathToData datasets/subsets/verb/vatex/vatex_all_centerK3_kmeans.json --nval 100

# P2 Finetuning
mkdir results/MSVD_S2
cp results/23d_z_msvd_opsedo_lr0.015_kl0.2_2022_06_10_23_02_42_P1K6Non/best_model.ckpt results/MSVD_S2
python scripts/train.py --di_rate 4 --dataset msvd --use_z --kl_rate 1.0 --non_gshuffle --learning_rate 0.001 --pathToData datasets/subsets/verb/msvd/annotations_msvd_K6_verb.json --resume_from results/MSVD_S2 --max_epoch 200 --use_diff_dists_loss --isphase2

cp results/23d_z_msrvtt_opsedo_lr0.015_kl0.5_cl1.0_2022_06_02_20_15_54_P1K5/best_model.ckpt results/MSRVTT_S2
python scripts/train.py --di_rate 4 --dataset msrvtt --use_z --kl_rate 1.0 --non_gshuffle --learning_rate 0.001 --pathToData datasets/subsets/verb/msrvtt/annotations_msrvtt_K5.json --resume_from results/MSRVTT_S2 --max_epoch 200 --use_diff_dists_loss --isphase2

cp vatex_P1K3/best_model.ckpt results/VATEX_S2
python scripts/train.py --di_rate 1.5 --dataset vatex --use_z --kl_rate 1.0 --non_gshuffle --learning_rate 0.001 --pathToData datasets/subsets/verb/vatex/annotations_vatex_K3json --resume_from results/VATEX_S2 --max_epoch 200 --use_diff_dists_loss --isphase2

# Evaluation
python scripts/eval.py --dataset msvd --use_z --mode test --use_z --val_num_samples 20 --mark test --mul_runs 1 --res_dir results/23d_z_msvd_opsedo_lr0.015_kl0.2_2022_06_10_23_02_42_P2di4

python scripts/eval.py --dataset msrvtt --use_z --mode test --use_z --val_num_samples 20 --mark test --mul_runs 1 --res_dir results/23d_z_msrvtt_opsedo_lr0.015_kl0.5_cl1.0_2022_06_02_20_15_54_P2di4