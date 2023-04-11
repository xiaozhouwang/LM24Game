from itertools import permutations, product, chain
from fractions import Fraction as F
from itertools import zip_longest
from re import sub, findall, search
import ast
from tqdm import tqdm
import pandas as pd


def solve(digits):
    """
    borrowed from https://rosettacode.org/wiki/24_game/Solve#Python
    """
    all_solutions = []
    
    digilen = len(digits)
    # length of an exp without brackets 
    exprlen = 2 * digilen - 1
    # permute all the digits
    digiperm = sorted(set(permutations(digits)))
    # All the possible operator combinations
    opcomb   = list(product('+-*/', repeat=digilen-1))
    # All the bracket insertion points:
    brackets = ( [()] + [(x,y)
                         for x in range(0, exprlen, 2)
                         for y in range(x+4, exprlen+2, 2)
                         if (x,y) != (0,exprlen+1)]
                 + [(0, 3+1, 4+2, 7+3)] ) # double brackets case
    for d in digiperm:
        for ops in opcomb:
            if '/' in ops:
                d2 = [('F(%s)' % i) for i in d] # Use Fractions for accuracy
            else:
                d2 = d
            ex = list(chain.from_iterable(zip_longest(d2, ops, fillvalue='')))
            for b in brackets:
                exp = ex[::]
                for insertpoint, bracket in zip(b, '()'*(len(b)//2)):
                    exp.insert(insertpoint, bracket)
                txt = ''.join(exp)
                try:
                    num = eval(txt)
                    if num == 24:
                        if '/' in ops:
                            exp = [ (term if not term.startswith('F(') else term[2])
                                    for term in exp ]
                        ans = ' '.join(exp).rstrip()
                        all_solutions.append(ans)
                except ZeroDivisionError:
                    continue
    solution_set = set([remove_unnecessary_parentheses(x).strip().replace(" ", "") for x in all_solutions])
    return list(solution_set)

def remove_unnecessary_parentheses(x):
    """
    e.g.
    (1+2)+3+4 --> 1+2+3+4
    """
    try:
        *n,=sub('\D','',x);x=sub('\d','9',x);v,i,r,l=eval(x),0,lambda d,a,s:d.replace(s,"?",a).replace(s,"",1).replace("?",s),lambda:len(findall('\(',x))
        while i<l():
            j=0
            while j<l():
                h=r(r(x,i,"("),j,")")
                try:
                    if eval(h)==v:i=j=-1;x=h;break
                except:0
                j+=1
            i+=1
        return sub('9','%s',x)%tuple(n)
    except ZeroDivisionError:
        return x


def is_valid_solution(answer, digits):
    allowed = set('() +-*/\t'+''.join(digits))
    ok = all(ch in allowed for ch in answer) and \
         all(digits.count(dig) == answer.count(dig) for dig in set(digits)) \
         and not search('\d\d', answer)
    if ok:
        try:
            ast.parse(answer)
        except:
            ok = False
    try:
        value_match = abs(eval(answer) - 24) < 1e-6
    except ZeroDivisionError:
        return False
    return ok and value_match


if __name__ == "__main__":
    all_digits = list(product(range(1, 10), repeat=4))
    data2store = []
    for digits in tqdm(all_digits):
        digits = [str(x) for x in digits]
        solutions = solve(digits)
        if len(solutions) == 0:
            solution = "" # no solution found
            data2store.append([' '.join(digits), solution])
        else:
            for solution in solutions:
                if is_valid_solution(solution, digits):
                    data2store.append([' '.join(digits), solution])
    df2store = pd.DataFrame(data2store, columns=['input', 'solution'])
    print(df2store.shape)
    df2store.to_parquet("24game_data.parquet")