#!usr/bin/env python
# -*- coding:utf-8 _*-
import numpy as np


def sub_mm2(l_trial, total_scores, face_scores, threshold_total, thershold_face):
    assert len(l_trial) == len(total_scores)
    f = open("subjective_results.txt", "w")
    count = 0
    mean_true = [0, 0]
    mean_false = [0, 0]
    for i in range(len(l_trial)):
        trg, utt_a, utt_b = l_trial[i].strip().split(' ')
        if trg == '1':
            mean_true[0] += total_scores[i]
            mean_true[1] += face_scores[i]
            if total_scores[i] > threshold_total and face_scores[i] < thershold_face:
                f.write(
                    "{}   total:{} score1:{}\n".format("|".join([trg, utt_a, utt_b]), total_scores[i], face_scores[i]))
                count += 1
        if trg == '0':
            mean_false[0] += total_scores[i]
            mean_false[1] += face_scores[i]
            if total_scores[i] < threshold_total and face_scores[i] > thershold_face:
                f.write(
                    "{}   total:{} score1:{}\n".format("|".join([trg, utt_a, utt_b]), total_scores[i], face_scores[i]))
                count += 1
    mean_true = np.asarray(mean_true) / len(l_trial) * 2
    mean_false = np.asarray(mean_false) / len(l_trial) * 2
    print("get {} subjective results, thershold=[{},{}], mean_true=[{},{}], mean_false=[{},{}]".format(count,
                                                                                                       threshold_total,
                                                                                                       thershold_face,
                                                                                                       mean_true[0],
                                                                                                       mean_true[1],
                                                                                                       mean_false[0],
                                                                                                       mean_false[1]))
    f.close()
