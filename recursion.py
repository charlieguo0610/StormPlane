


dic = {}
enemy_list = [[1, 100], [2, 200]]
l = []
def knapSack(hp, n):
    if hp == 0 or n == 0:
        return 0
    key = str(hp) + "#" + str(n)

    if key in dic:
        return dic[key]

    if enemy_list[n - 1][0] > hp:
        dic[key] = knapSack(hp, n - 1)
        return dic[key]

    else:
        choose = enemy_list[n-1][1] + knapSack(hp-enemy_list[n-1][0], n-1)
        reject = knapSack(hp, n-1)
        if choose > reject:
            l.append((enemy_list[n-1][0], enemy_list[n-1][1]))
            dic[key] = choose
        else:
            dic[key] = reject


        # dic[key] = max(enemy_list[n - 1][1] + knapSack(hp - enemy_list[n - 1][0], n - 1),
        #                knapSack(hp, n - 1))
        return dic[key]

print(knapSack(4, len(enemy_list)))
print(dic)