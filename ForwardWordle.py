import numpy as np
import pandas as pd
import pickle
import time
import copy
from sklearn.linear_model import LogisticRegression


with open('wlist.txt', 'r+') as file:
    lines = file.readlines()
    r5 = [r[:5] for r in lines]

def log(x):
    if x > 0:
        return np.log(x)
    else:
        return 0



class option_branch:
    def __init__(self, current, word, step=0, result=''):
        self.current = current
        self.root = not bool(step)
        self.step = step
        self.result = result
        self.word = word
        if step != 5:
            self.let = word[step]
        else:
            self.let = ''
        
    def forward(self):
        if len(self.current) == 0:
            return []
        self.b = []
        self.y = []
        self.g = []
        for c in self.current:
            if c[self.step] == self.let:
                self.g.append(c)
            elif self.let in c:
                self.y.append(c)
            else:
                self.b.append(c)
        self.b = option_branch(self.b, self.word, step=self.step+1, result=self.result+'b')
        self.y = option_branch(self.y, self.word, step=self.step+1, result=self.result+'y')
        self.g = option_branch(self.g, self.word, step=self.step+1, result=self.result+'g')
        return [self.b, self.y, self.g]



class option_tree:
    def __init__(self, current, word):
        self.current = current
        self.word = word

    def tally(self):
        branches = [option_branch(self.current, self.word)]
        for step in range(5):
            new_branches = []
            for b in branches:
                x = b.forward()
                if len(x) > 0:
                    new_branches.extend(x)
            branches = new_branches
        return [(b.result,len(b.current)) for b in branches]

    def entropy(self):
        tal = self.tally()
        N = len(self.current)
        return sum([- t[1] / N * log(t[1] / N) for t in tal]) / np.log(2)


class base_game:
    def __init__(self):
        self.w = list('abcdefghijklmnopqrstuvwxyz')
        self.g = {}
        self.y = {}
        self.b = []
        self.options = r5.copy()
        for k in range(5):
            self.y[k] = []
            self.g[k] = []
        self.turns = 0
        self.solved = False
        self.re = np.log(len(self.options)) / np.log(2)

    def get_options(self):
        remove = []
        yellows = []
        for i in range(5):
            yellows = yellows + self.y[i]
        yellows = list(set(yellows))
        self.yyy = yellows
        good = []
        for o in self.options:
            keep = True
            for k in range(5):
                if len(self.g[k]) >= 1:
                    if self.g[k][0] != o[k]:
                        keep = False
                        break
                if len(self.y[k]) >= 1:
                    if o[k] in self.y[k]:
                        keep = False
                        break
                if o[k] not in self.w:
                    keep = False
                    break
            if not keep:
                continue
            else:
                if np.all([ly in o for ly in yellows]):
                    good.append(o)
        self.options = good

    def check(self, guess, res):
        for k in range(5):
            if res[k] == 'y':
                self.y[k].append(guess[k])
            if res[k] == 'g':
                self.g[k].append(guess[k])
            if res[k] == 'b':
                if guess[k] in self.w:
                    self.w.remove(guess[k])
        self.turns += 1
        print(guess)
        self.bw = guess

    def set_ans(self, ans):
        self.ans = ans

    def autocheck(self, guess):
        res = ''
        for k in range(5):
            if guess[k] == self.ans[k]:
                res += 'g'
            elif guess[k] in self.ans:
                res += 'y'
            else:
                res += 'b'
        self.check(guess, res)

    def guess(self, full=False, verbose=False):
        self.re = np.log(len(self.options)) / np.log(2)
        if full and self.re > 0:
            option_list = r5
        else:
            option_list = self.options
        ms = -1.0
        for word in option_list:
            e = option_tree(self.options, word).entropy()
            if e > ms:
                ms = e
                bw = word
        self.ms = ms
        self.autocheck(bw)
        self.get_options()
        self.bw = bw
        if verbose:
            print(self.turns, bw, round(e, 4), round(self.re, 4))
        if self.bw == self.ans:
            self.solved = True

    def guess_with_model(self, mod, thres=0.5):
        self.re = np.log(len(self.options)) / np.log(2)
        ms1 = -1.0
        if self.re > 0.0:
            option_list = r5
        else:
            option_list = self.options
        for word in option_list:
            e1 = option_tree(self.options, word).entropy()
            if e1 > ms1:
                ms1 = e1
                bw1 = word
        ms2 = -1.0   
        for word in self.options:
            e2 = option_tree(self.options, word).entropy()
            if e2 > ms2:
                ms2 = e2
                bw2 = word
        dec = mod.predict(np.array([self.re, ms1, ms2, self.turns]).reshape((1,4)))
        if dec > thres:
            self.ms = ms1
            self.autocheck(bw1)
            self.get_options()
            self.bw = bw1
        else:
            self.ms = ms2
            self.autocheck(bw2)
            self.get_options()
            self.bw = bw2
        print(self.bw)
        if self.bw == self.ans:
            self.solved = True


    def solve(self, verbose=False):
        bw = 'raise'
        T = option_tree(r5, 'raise')
        self.autocheck('raise')
        self.get_options()
        e = T.entropy()
        re = np.log(len(self.options)) / np.log(2)
        while not self.solved:
            self.guess(verbose=verbose)
        return self.turns

    def solve_with_model(self, mod, thres=0.5):
        bw = 'raise'
        T = option_tree(r5, 'raise')
        self.autocheck('raise')
        self.get_options()
        e = T.entropy()
        re = np.log(len(self.options)) / np.log(2)
        while not self.solved:
            self.guess_with_model(mod=mod, thres=thres)
        return self.turns


class decision_branch:
    def __init__(self, game, decisions='', data=[]):
        self.game = game
        self.decisions = decisions
        self.data = copy.deepcopy(data)

    def forward(self):
        if self.game.solved:
            return [self]
        self.outside = copy.deepcopy(self.game)
        self.outside.guess(full=True)
        self.inside = copy.deepcopy(self.game)
        self.inside.guess(full=False)
        self.data.append([np.log(len(self.game.options)) / np.log(2), copy.copy(self.outside.ms), copy.copy(self.inside.ms), self.game.turns])
        if self.outside.ms == self.inside.ms:
            return [decision_branch(self.inside, decisions=self.decisions+'i', data=self.data)]
        results = []
        if self.inside.bw != self.game.bw:
            results.append(decision_branch(self.inside, decisions=self.decisions+'i', data=self.data))
        if self.outside.bw != self.game.bw:
            results.append(decision_branch(self.outside, decisions=self.decisions+'o', data=self.data))
        return results

class decision_tree:
    def __init__(self, answer, warm=True):
        self.game = base_game()
        self.game.set_ans(answer)
        if warm:
            self.game.autocheck('raise')
            self.game.get_options()

    def tally(self):
        self.branches = [decision_branch(self.game)]
        all_solved = False
        while not all_solved:
            leaves = []
            for branch in self.branches:
                leaf = branch.forward()
                leaves.extend(leaf)
            self.branches = leaves
            all_solved = np.all([b.game.solved for b in self.branches])
        result = []
        data = []
        for leaf in self.branches:
            result.append((leaf.decisions, leaf.game.turns))
            data.append((leaf.decisions, leaf.data))
        return dict(result), dict(data)


# G = base_game()
# G.check('raise', 'bbybg')
# G.get_options()
# T = sorted([(r, option_tree(G.options, r).entropy()) for r in G.options], key=lambda x: -x[-1])
# print(T[:5]) 
# Do this ^

