"""
For generating extensive games for Gambit
"""

import numpy as np
import os

# set active directory to location of this file
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def gen_id(values):
    """Helper that generates a unique integer id from a list of boolean values"""
    return sum((2**i * y for i, y in enumerate(values)))


class Node:

    def write(self, lines: list):
        raise NotImplementedError


class Terminal(Node):

    def __init__(self, xs: np.ndarray, ys: tuple, u: np.ndarray, v: np.ndarray):
        self.xs = xs
        self.ys = ys
        self.u = u
        self.v = v

    def write(self, lines: list):
        my_id = 2**len(self.ys) * gen_id(self.xs) + gen_id(self.ys) + 1
        avg = np.mean(self.ys)
        tie = (avg == 0.5)
        outcome = (avg > 0.5)
        if tie:
            payoffs_iter = (
                str((u + v) * 0.5) for u, v in zip(self.u, self.v)
            )
        else:
            payoffs_iter = (
                str(u * (x == outcome) + v * (y == outcome))
                for x, y, u, v in zip(
                    self.xs, self.ys, self.u, self.v
                )
            )
        payoffs = '{ ' + ' '.join(payoffs_iter) + ' }'
        lines.append(f't "" {my_id} "" {payoffs}')


class PlayerVote(Node):

    def __init__(
        self,
        n_players: int,
        xs: np.ndarray,
        ys: tuple,
        u: np.ndarray,
        v: np.ndarray
    ):
        self.xs = xs
        self.player = len(ys)
        self.ys = ys
        if self.player == n_players - 1:
            self.children = [
                Terminal(xs, ys+(False,), u, v),
                Terminal(xs, ys+(True,), u, v)
            ]
        else:
            self.children = [
                PlayerVote(n_players, xs, ys+(False,), u, v),
                PlayerVote(n_players, xs, ys+(True,), u, v)
            ]
    
    def write(self, lines: list):
        # use the power of binary to get a unique infoset number
        infoset = 2 * gen_id(self.ys) + self.xs[self.player] + 1
        options = f'{{ "y{self.player+1}=a" "y{self.player+1}=b" }}'
        lines.append(f'p "" {self.player + 1} {infoset} "" {options} 0')
        for child in self.children:
            child.write(lines)


class Head(Node):

    def __init__(self, n_players: int = 2, p: np.ndarray = None, u: np.ndarray = None, v: np.ndarray = None):
        if p is None:
            self.p = 0.5 * np.ones(n_players)
        else:
            assert p.shape == (n_players,)
            self.p = p
        if u is None:
            self.u = np.ones(n_players)
        else:
            assert u.shape == (n_players,)
            self.u = u
        if v is None:
            self.v = np.ones(n_players)
        else:
            assert v.shape == (n_players,)
            self.v = v
        self.n_players = n_players
        
        self.xs = self._create_xs()
        self.children = [PlayerVote(self.n_players, x, tuple(), self.u, self.v) for x in self.xs]
    
    def _create_xs(self):
        binary_nums = [bin(i)[2:] for i in range(2**self.n_players)]
        binary_nums = [
            '0'*(self.n_players - len(num)) + num
            for num in binary_nums
        ]
        return np.array(
            [
                [int(x) for x in binary_num] for binary_num in binary_nums
            ],
            dtype=bool
        )
    
    def write(self, lines: list):
        state_names = [
            ', '.join((f'x{i+1}={"b" if val else "a"}' for i, val in enumerate(row)))
            for row in self.xs
        ]
        probas = [
            np.prod([p if val else 1-p for p, val in zip(self.p, row)])
            for row in self.xs
        ]
        options = '{ ' + ' '.join((
            f'"{name}" {proba}' for name, proba in zip(state_names, probas)
        )) + ' }'
        lines.append(f'c "" 1 "" {options} 0')
        for child in self.children:
            child.write(lines)


def write_game(
    n_players: int = 2,
    p: np.ndarray = None,
    u: np.ndarray = None,
    v: np.ndarray = None,
    filename: str = None
):
    head = Head(n_players, p, u, v)
    lines = []
    head.write(lines)
    player_enum = '{ ' + ' '.join((f'"Player {i+1}"' for i in range(n_players))) + ' }'
    header = f'EFG 2 R "{n_players} player voting game" {player_enum}\n'
    if filename is None:
        filename = f'game_{n_players}_players.efg'
    with open(filename, 'w') as fh:
        fh.write(header + '\n'.join(lines))


if __name__ == '__main__':
    write_game(4)
