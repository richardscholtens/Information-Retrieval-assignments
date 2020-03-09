#!/usr/bin/python3
#
# student: J.F.P. (Richard) Scholtens
# studentnr.: s2956586
from operator import itemgetter
from collections import defaultdict


def matrix(file):
    """File to two nested dictionary."""
    dic = defaultdict(set)
    for line in file:
        line = line.split()
        dic[line[0]].add(line[1])
    return dic


def file_to_list(file):
    lst = []
    for line in file:
        line = line.rstrip()
        user, vip = line.split()
        lst.append((user, vip))
    return lst


def main():
    # Saves all Test users - Followers in dictionary.
    with open('Twitter2014/matrix_training.csv', 'r') as file:
        pair_dic = matrix(file)

    # Saves all Training users - Should follow in dictionary.
    with open('Twitter2014/matrix_test.csv', 'r') as file:
        test_lst = file_to_list(file)

    # Saves all Users - Similiarity scores in dictionary.
    sim_dic = {}
    with open('Twitter2014/user_similarity.csv', 'r') as user_similarity:
        for line in user_similarity:
            items = line.split()
            if items[0] not in sim_dic.keys():
                sim_dic[items[0]] = [(items[1], float(items[2]))]
                sim_dic[items[1]] = [(items[0], float(items[2]))]
            else:
                sim_dic[items[0]].append((items[1], float(items[2])))
                if items[1] not in sim_dic.keys():
                    sim_dic[items[1]] = [(items[0], float(items[2]))]
                else:
                    sim_dic[items[1]].append((items[0], float(items[2])))

    # Saves all Users - Top 10 followers
    top10_dic = {}
    for key, val in sim_dic.items():
        top10_dic[key] = sorted(val, key=itemgetter(1), reverse=True)[:10]

    recommend_dic = {}

    # Save Test Users - Top 10 follower in dictionary.
    for user, vip in test_lst:
        top10_lst = top10_dic[user]
        user_follows = set(pair_dic[user])
        prediction_dic = defaultdict(int)
        for tup in top10_lst:
            top10_follows = set(pair_dic[tup[0]])
            intersect = user_follows & top10_follows
            no_follow = top10_follows - intersect
            for no_f in no_follow:
                prediction_dic[no_f] += tup[1]
        sorted_pred = sorted(prediction_dic.items(),
                             key=itemgetter(1), reverse=True)[:10]
        recommend_dic[user] = [i[0] for i in sorted_pred]

    # Evaluates Test users should follow - Recommend followers
    total = 0
    with open("2.out", "w+") as f:
        for user, vip in test_lst:
            score = 0
            if vip in recommend_dic[user]:
                score = 1
                total += 1
            f.write('{0} {1} {2} {3}\n'.format(user,
                    vip, recommend_dic[user], score))
        f.write("Total matches: {0}".format(total))
    print(total)

if __name__ == '__main__':
    main()
