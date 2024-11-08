

#


#python main.py -O --mask_image data/lty/li --text "a letter '李' with rope style" --workspace sup5_li_rope --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/mu --text "a letter '木' with rope style" --workspace sup5_mu_rope --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/zi --text "a letter '子' with rope style" --workspace sup5_zi_rope1 --vram_O --iters 10000 --mask_stop_step 5000 --update_mask_epoch 50 --mask_use_step 8000;
#python main.py -O --mask_image data/lty/yang --text "a letter '羊' with rope style" --workspace sup5_yang_rope --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/kun --text "a letter '困' with rope style" --workspace sup5_kun_rope --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/ha --text "a letter '哈' with rope style" --workspace sup5_ha_rope --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/de --text "a letter '的' with rope style" --workspace sup5_de_rope --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/di --text "a letter '地' with rope style" --workspace sup5_di_rope --vram_O --iters 10000 --mask_stop_step 8000;






#python main.py -O --mask_image data/lty/li --text "a letter '李' with green bamboo style" --workspace sup5_li_bamboo1 --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/mu --text "a letter '木' with green bamboo style" --workspace sup5_mu_bamboo1 --vram_O --iters 10000 --mask_stop_step 8000;
python main.py -O --mask_image data/lty/zi --text "a letter '子' with green bamboo style" --workspace sup5_zi_bamboo2 --vram_O --iters 15000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/yang --text "a letter '羊' with green bamboo style" --workspace sup5_yang_bamboo1 --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/kun --text "a letter '困' with green bamboo style" --workspace sup5_kun_bamboo1 --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/ha --text "a letter '哈' with green bamboo style" --workspace sup5_ha_bamboo1 --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/de --text "a letter '的' with green bamboo style" --workspace sup5_de_bamboo1 --vram_O --iters 10000 --mask_stop_step 8000;
#python main.py -O --mask_image data/lty/di --text "a letter '地' with green bamboo style" --workspace sup5_di_bamboo1 --vram_O --iters 10000 --mask_stop_step 8000;

















#python main.py -O --mask_image data/English_Letter_blur/a2+_mask --text "a letter 'A' with banana style, red flower on the bananas" --workspace A_banana_top5 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40



#python main.py --mask_image data/English_Letter_blur/a2+_mask -O --text "a letter 'A' with banana style, a red flower on the top of bananas" --workspace trial_ablation31 --iters 20000 --known_view_interval 24000 --IF --batch_size 1 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 400;


#python main.py -O --mask_image data/English_Letter_blur/5 --text "a letter 'A' with banana style, red flower in the middle of bananas" --workspace A_banana_nomask --iters 5000 --known_view_interval 240000 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40
#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a letter 'A' with banana style" --workspace 成功trial_A32 --iters 5000 --known_view_interval 240000 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40
#
#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a letter 'A' with banana style" --workspace trial_if_mask8+_A128_banana128-5000-256-10000 --iters 5000 --known_view_interval 240000 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch





#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with Van Gogh style" --workspace trial_Christmas_S --iters 13000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40

#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with Van Gogh style" --workspace 成功trial_huo3 --iters 13000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40

#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with Van Gogh style" --workspace trial_1289 --iters 13000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40



#python main.py -O --mask_image data/English_Letter_blur/5 --text "a number '5' with hay style" --workspace trial_sup_511 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#
#python main.py -O --mask_image data/English_Letter_blur/5 --text "a number '5' with Super Mario style" --workspace trial_sup_512 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;


#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with winter style" --workspace trial_sup_51 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with wash painting style" --workspace trial_sup_54 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with ink style" --workspace trial_sup_55 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with Van Gogh style" --workspace trial_sup_56 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with birthday style" --workspace trial_sup_57 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with Christmas style" --workspace trial_sup_58 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/5 --text "a number '5' with happy style" --workspace trial_sup_59 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;





#
#
#python main.py --test -O --mask_image data/English_Letter_blur/S --text "a letter 'S' with strawberry style" --workspace trial_sup_S1 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/S --text "a letter 'S' with apple style" --workspace trial_sup_S2 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/K --text "a letter 'K' with strawberry style" --workspace trial_sup_K1 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/K --text "a letter 'K' with apple style" --workspace trial_sup_K2 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/M --text "a letter 'M' with feather style" --workspace trial_sup_M1 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/M --text "a letter 'M' with colorful feather style" --workspace trial_sup_M2 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/A --text "a letter 'A' with feather style" --workspace trial_sup_A1 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/A --text "a letter 'A' with colorful feather style" --workspace trial_sup_A2 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#
#python main.py --test -O --mask_image data/English_Letter_blur/A --text "a letter 'A' with water style" --workspace trial_sup_A3 --iters 13000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/A --text "a letter 'A' with colorful blood style" --workspace trial_sup_A4 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#
#
#python main.py --test -O --mask_image data/English_Letter_blur/shui --text "a word '水' with water style" --workspace trial_sup_shui1 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/shui --text "a letter '水' with colorful blood style" --workspace trial_sup_shui2 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/tu --text "a letter '土' with colorful sand style" --workspace trial_sup_tu1 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;
#python main.py --test -O --mask_image data/English_Letter_blur/tu --text "a letter '土' with colorful sandbeach style" --workspace trial_sup_tu2 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 8000 --update_mask_epoch 40;


#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_ropeA_sup30 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 3000 --mask_use_step 9000 --update_mask_epoch 30;
#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_ropeA_sup20 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 2000 --mask_use_step 9000 --update_mask_epoch 20;
#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_ropeA_sup10 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 1000 --mask_use_step 9000 --update_mask_epoch 10;




#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_ropeA_sup40 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 9000 --update_mask_epoch 40;
#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_ropeA_sup50 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 9000 --update_mask_epoch 50;
#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_ropeA_sup60  --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 9000 --update_mask_epoch 60;
#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_ropeA_sup70  --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 9000 --update_mask_epoch 70;
#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_ropeA_sup80  --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 9000 --update_mask_epoch 80;
#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_ropeA_sup90  --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 9000 --update_mask_epoch 90;





#python main.py -O --mask_image data/mask_image_blur/雅珠/F --text "style" --workspace trial_ropeF1 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 8000 --mask_use_step 8000 --update_mask_epoch 4000;
#python main.py -O --mask_image data/mask_image_blur/兰亭圆3/F --text "a letter 'F' with rope style" --workspace trial_ropeF2 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 5000 --mask_use_step 8000 --update_mask_epoch 4000;

#python main.py -O --mask_image data/mask_image_blur/雅珠/G --text "rope style" --workspace trial_ropeG1 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 8000 --mask_use_step 8000 --update_mask_epoch 4000;
#python main.py -O --mask_image data/mask_image_blur/兰亭圆3/G --text "a letter 'G' with rope stydabfrab e" --workspace trial_ropeG2 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 5000 --mask_use_step 8000 --update_mask_epoch 4000;

#python main.py -O --mask_image data/mask_image_blur/雅珠/H --text "rope style" --workspace trial_ropeH1 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 8000 --mask_use_step 8000 --update_mask_epoch 4000;
#python main.py -O --mask_image data/mask_image_blur/兰亭圆3/H --text "a letter 'H' with rope style" --workspace trial_ropeH2 --iters 10000 --known_view_interval 24 --vram_O --mask_stop_step 5000 --mask_use_step 8000 --update_mask_epoch 4000;


#python main.py -O --mask_image data/mask_image_blur/雅珠/E --text "a letter 'E' with rope style" --workspace trial_ropeE1 --iters 12000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 10000 --update_mask_epoch 40
#python main.py -O --mask_image data/mask_image_blur/兰亭圆2/E --text "a letter 'E' with rope style" --workspace trial_ropeE2 --iters 12000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 10000 --update_mask_epoch 40
#
#python main.py -O --mask_image data/mask_image_blur/雅珠/F --text "a letter 'F' with rope style" --workspace trial_ropeF1 --iters 12000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 10000 --update_mask_epoch 40
#python main.py -O --mask_image data/mask_image_blur/兰亭圆2/F --text "a letter 'F' with rope style" --workspace trial_ropeF2 --iters 12000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 10000 --update_mask_epoch 40
#
#python main.py -O --mask_image data/mask_image_blur/雅珠/G --text "a letter 'G' with rope style" --workspace trial_ropeG1 --iters 12000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 10000 --update_mask_epoch 40
#python main.py -O --mask_image data/mask_image_blur/兰亭圆2/G --text "a letter 'G' with rope style" --workspace trial_ropeG2 --iters 12000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 10000 --update_mask_epoch 40
#
#python main.py -O --mask_image data/mask_image_blur/雅珠/H --text "a letter 'H' with rope style" --workspace trial_ropeH1 --iters 12000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 10000 --update_mask_epoch 40
#python main.py -O --mask_image data/mask_image_blur/兰亭圆2/H --text "a letter 'H' with rope style" --workspace trial_ropeH2 --iters 12000 --known_view_interval 24 --vram_O --mask_stop_step 4000 --mask_use_step 10000 --update_mask_epoch 40

#python main.py -O --mask_image data/mask_image_blur/功夫/A --text "a letter 'A' with rope style" --workspace trial_A_rope21 --iters 5000  --known_view_interval 24 --vram_O --mask_stop_step 4000000 --mask_use_step 4000000 --update_mask_epoch 400









#python main.py --iters 5000 --mask_image data/English_Letter_blur/a3+_mask -O --text "a letter 'A' with banana style, a red flower in the middle of two bananas" --workspace trial_A3 --known_view_interval 2400000 --IF --vram_O
#python main.py --mask_image data/English_Letter_blur/huo -O --text "a word '火' with fire style" --workspace trial_nomask_huo3 --IF --vram_O --iters 10000;
#
#
#python main.py --mask_image data/English_Letter_blur/\$ -O --text "a symbol '$' with gold style" --workspace trial_nomask_\$ --IF --vram_O --iters 10000;
#python main.py --mask_image data/English_Letter_blur/9 -O --text "a number '9' with wood style" --workspace trial_nomask_9 --IF --vram_O --iters 10000;
#python main.py --mask_image data/English_Letter_blur/9 -O --text "a letter 'A' with banana style" --workspace trial_nomask_A --IF --vram_O --iters 10000;
#
#python main.py -O --IF --vram_O --mask_image data/English_Letter_blur/tian_mask --workspace trial_nomasktian2 --text "a cucumber on the top of some bananas."
#python main.py -O --IF --vram_O --mask_image data/English_Letter_blur/Q_mask --workspace trial_nomaskQ --text "a letter 'Q' with rope style, a banana is under the rope."
#python main.py -O --IF --vram_O --mask_image data/English_Letter_blur/o_mask --workspace trial_nomasko1 --text "two apples and a banana inside a rope, they look like a smiling face.";
#python main.py -O --IF --vram_O --mask_image data/English_Letter_blur/Hs_mask --workspace trial_nomaskHs2 --text "a rope is between a wood and a bamboo.";
#python main.py --mask_image data/English_Letter_blur/a3+_mask -O --text "a letter 'A' with banana style, a red flower in the middle of two bananas." --workspace trial_nomaskA1 --IF --vram_O
#python main.py --mask_image data/English_Letter_blur/a1+_mask -O --text "a letter 'A' with banana style, a red flower on the top of two bananas." --workspace trial_nomaskA2 --IF --vram_O
#python main.py -O --IF --vram_O --mask_image data/English_Letter_blur/!_mask --workspace trial_nomask! --text "a symbol '!' with carrot style, the carrot is on the top of an apple.";
#python main.py -O --IF --vram_O --mask_image data/English_Letter_blur/3_mask --workspace trial_nomask3 --text "a carrot is between a banana and a cucumber, the carrot on the top of the banana.";