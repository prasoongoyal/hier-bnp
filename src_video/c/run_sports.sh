for l in 3 4; do
  for b in 2 3; do
    nohup unbuffer python levelwise_gibbs_sampling_wrapper.py ~/sports5_frame_features_sorted.txt 1.0 0.1 1.0 /scratch/pg1338/tmp_files_sports/tmp"${l}""${b}" "${l}" "${b}" /scratch/pg1338/tmp_files_sports/all_beta_sports5_a1.0_s0.1_g1.0_l"${l}"_b"${b}"_temporal_switch.txt > /scratch/pg1338/tmp_files_sports/sports5_a1.0_s0.1_g1.0_l"${l}"_b"${b}"_temporal_switch_output.txt &
  done
done
