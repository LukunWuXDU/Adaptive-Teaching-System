#!/bin/bash
# cd /home/xxxxxxxx/Adaptive-Teaching-System-main

## EEG
# for subject in {1..10}
# do
#   python preprocess/process_eeg_whiten.py --subject $subject
# done

## MEG
for subject in {1..4}
do
  python preprocess/process_meg.py --subject $subject
done


# python preprocess/process_resize.py --type eeg
# python preprocess/process_resize.py --type meg
