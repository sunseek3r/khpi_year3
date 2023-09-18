import numpy as np

n_princes = 12


def optimal_stopping():
    """
    Ми проходимось по перших k = n/e принцам. Потім, з тіх що залишилися, вибираємо першого, кращого за всіх
    попередніх. Така стратегія дає приблизно 1/e ймовірність перемоги.
    :return:
    """
    # each prince has random "goodness"
    # princes = np.random.normal(loc=0.0, scale=1.0, size=n_princes)
    # the princes have goodness from 0 to n_princes
    princes = np.random.permutation(n_princes)
    # print(princes)
    true_win = max(princes)
    k = int(n_princes / np.e)
    max_k = -1e9

    for i in range(k):
        if princes[i] > max_k:
            max_k = princes[i]
    # print(max_k)
    for i in range(k, n_princes):
        if princes[i] > max_k:
            max_k = princes[i]
            break

        if i + 1 == n_princes:
            max_k = princes[i]
    # print(max_k, true_win)
    return true_win == max_k


def dynamic():
    """
    Є наступна стратегія:
    Приходить і-й принц. Якщо він найкращій серед тих, що попались раніше, ймовірність вибрати його -
    і / кількість переглянутих принців. Найгірший випадок - найкращий принц стоїть близько до пачатку.
    Якщо ж ніхто не був вибраний, то вибираємо останнього.
    Така стратегія дає ~30%.
    :return:
    """
    princes = np.random.permutation(n_princes)
    dynamic_rating = []

    chosen = -5
    for ind, val in enumerate(princes):
        dynamic_rating.append(val)
        dynamic_rating.sort()
        if val == dynamic_rating[-1]:
            win_prob = (ind + 1) / n_princes

            check = np.random.uniform(size=1)
            if check < win_prob:
                chosen = val
                break

    if chosen == -5:
        chosen = princes[-1]

    return chosen == n_princes - 1


def check_each():
    """
    Кожен i-й принц має шанс 1/(n - i) i=0..n-1.
    Найгірша стратегія, дає ~1/12 шанс, що зрівняно з вгадуванням.
    :return:
    """
    princes = np.random.permutation(n_princes)

    chosen = -5
    for ind, val in enumerate(princes):
        cur_chance = 1 / (n_princes - ind)

        check = np.random.uniform(size=1)

        if check < cur_chance:
            chosen = val
            break
    if chosen == -5:
        chosen = princes[-1]

    return chosen == n_princes - 1


def experiment(strategy_name, strategy_func):
    num_experiments = 100000
    win = 0
    for i in range(num_experiments):
        win += strategy_func()
    print (f"{strategy_name} got {win * 100 / num_experiments}% win")


if __name__ == '__main__':
    experiment("Optimal stopping", optimal_stopping)
    experiment("Dynamic probability", dynamic)
    experiment("Check each", check_each)
