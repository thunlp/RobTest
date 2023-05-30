#!bash
data=('sst2' 'jigsaw' 'agnews') 
attacker=(['typo', 'glyph', 'phonetic ', 'synonym', 'contextual', 'inflect', 'syntax', 'distraction'])
victim_model=('roberta-base', 'roberta-large')
dis_type=('char','word')
mode=('rule', 'score')

for i in "${data[@]}"
do
    for j in "${attacker[@]}"
    do
        for k in "${victim_model[@]}"
        do
            for l in "${dis_type[@]}"
            do
                for m in "${mode[@]}"
                do
                    python src/robtest.py --mode $m --attacker $j --data $i --dis_type $l --choice both --victim_model $k
                done
            done
        done
    done
done