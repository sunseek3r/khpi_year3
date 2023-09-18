import numpy as np
import scipy
num_tournaments = 100000


def tournament(a: int, b: int, n: int):
    """
    calculates number of "good" tournaments i.e. in which a games won out of b games
    :param a:
    number of wins needed
    :param b:
    number of games in tournament
    :param n:
    number of tournaments
    :return:
    number of "good" tournaments
    """

    # we suppose chance of winning is 50%
    tournaments = np.random.uniform(size=(n, b)) < 0.5  # suppose each True value is a win and False is a loss
    tournaments = np.sum(tournaments, axis=1).reshape(-1, 1)

    result = np.sum(tournaments == a)
    return result


if __name__ == "__main__":
    a1, b1 = 2, 3
    a2, b2 = 3, 5
    tournament1 = tournament(a1, b1, num_tournaments)
    tournament2 = tournament(a2, b2, num_tournaments)

    if tournament1 > tournament2:
        print(f"There's better chance to win {a1} out of {b1} games.")
    else:
        print(f"There's better chance to win {a2} out of {b2} games.")

    print(f"Probabilities are {tournament1 / num_tournaments} and {tournament2 / num_tournaments}")
    prob1 = scipy.special.comb(b1, a1) * (0.5 ** a1) * (0.5 ** (b1 - a1))
    prob2 = scipy.special.comb(b2, a2) * (0.5 ** a2) * (0.5 ** (b2 - a2))
    print(f"According to Bernoulli formula the probabilities are {prob1} and {prob2}")