### Problem 1

# Say "Hello, World!" With Python

print("Hello, World!")

# Python If-Else

n = int(input())

if n%2:
    print("Weird")
elif 2 <= n <= 5:
    print("Not Weird")
elif 6 <= n <= 20:
    print("Weird")
else:
    print("Not Weird")

# Arithmetic Operators

a = int(input())
b = int(input())
print(a+b)
print(a-b)
print(a*b)

# Python: Division

a = int(input())
b = int(input())
print(a//b)
print(a/b)

# Loops

n = int(input())
for i in range(n):
    print(i*i)

# Write a function

def is_leap(year):
    if year%4:
        return False
    if not year%400:
        return True
    if not year%100:
        return False
    return True

year = int(input())
print(is_leap(year))

# Print Function

n = int(input())
for i in range(1,n+1):
    print(i,end="")

### Data types

# List Comprehensions

x = int(input())
y = int(input())
z = int(input())
n = int(input())
print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k != n])

# Find the Runner-Up Score!

n = int(input())
arr = map(int, input().split())

"""Take the second-highest score (sort the list -> sorted yields ascending order -> take second-to-last element)"""
runner_up_score = sorted(list(set(arr)))[-2]

print(runner_up_score)

# Nested Lists

scores = []

for _ in range(int(input())):
    name = input()
    score = float(input())
    scores.append((name, score))
    
second_lowest_score = sorted(set(score for name,score in scores))[1]
students_with_second_lowest_score = sorted([name for name,score in scores if score == second_lowest_score])

for name in students_with_second_lowest_score:
    print(name)

# Finding the percentage

n = int(input())
student_marks = {}
for _ in range(n):
    name, *line = input().split()
    scores = list(map(float, line))
    student_marks[name] = scores
query_name = input()

average_mark = sum(student_marks[query_name])/len(student_marks[query_name])
print(f"{average_mark:.2f}")

# Lists

l = []
n = int(input())
for i in range(n):
    tokens = input().split()
    command = tokens[0]
    if command == "append":
        l.append(int(tokens[1]))
    elif command == "insert":
        l.insert(int(tokens[1]), int(tokens[2]))
    elif command == "sort":
        l.sort()
    elif command == "remove":
        l.remove(int(tokens[1]))
    elif command == "pop":
        l.pop()
    elif command == "print":
        print(l)
    else:
        l.reverse()

# Tuples

"""Read and discard n, useless since values are on the same row"""
raw_input() 
print(hash(tuple(map(int, raw_input().split()))))

### Strings

# sWAP cASE

def swap_case(s):
    result = []
    for char in s:
        if char.islower():
            result.append(char.upper())
        elif char.isupper():
            result.append(char.lower())
        else:
            result.append(char)
    return "".join(result)

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

# String Split and Join

def split_and_join(line):
    return "-".join(line.split(" "))

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?

def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

# Mutations

def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

# Find a string

def count_substring(string, sub_string):
    return sum(1 if string[start:start+len(sub_string)] == sub_string else 0 for start in range(len(string)))

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

# String Validators

import re

def print_whether_ok(regex, string):
    print(re.search(re.compile(regex), string) is not None)

s = input()

print_whether_ok("[a-zA-Z0-9]", s)
print_whether_ok("[a-zA-Z]", s)
print_whether_ok("[0-9]", s)
print_whether_ok("[a-z]", s)
print_whether_ok("[A-Z]", s)

# Text Alignment

"""Replace all ______ with rjust, ljust or center."""

thickness = int(input()) #This must be an odd number
c = 'H'

"""Top Cone"""
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

"""Top Pillars"""
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

"""Middle Belt"""
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

"""Bottom Pillars"""
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

"""Bottom Cone"""
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap

import textwrap

def wrap(string, max_width):
    return "\n".join(textwrap.wrap(string, max_width))

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# Designer Door Mat

n = int(input().split()[0])
m = n*3

for i in range(n//2):
    print((".|."*(1+2*i)).center(m, '-'))
print("WELCOME".center(m, '-'))
for i in range(n//2):
    print((".|."*(2*(n//2-i)-1)).center(m, '-'))

# String Formatting

def print_formatted(number):
    lines = [[str(i), oct(i)[2:], hex(i)[2:].upper(), bin(i)[2:]] for i in range(1,number+1)]
    width = len(lines[-1][-1])
    for line in lines:
        print(" ".join(token.rjust(width, " ") for token in line))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

# Alphabet Rangoli

from string import ascii_lowercase

def line_content(char_index, recursions):
    char = ascii_lowercase[char_index]
    if recursions == 0:
        return char
    return f'{char}-{line_content(char_index-1, recursions-1)}-{char}'

def print_rangoli(size):
    line_width = 4*(size-1)+1
    for i in range(size):
        print(line_content(size-1, i).center(line_width, '-'))
    for i in reversed(range(size-1)):
        print(line_content(size-1, i).center(line_width, '-'))

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

# Capitalize!

#!/bin/python3

import math
import os
import random
import re
import sys

"""Complete the solve function below."""
def solve(s):
    return " ".join([token[0].upper()+token[1:] if len(token) else "" for token in s.split(" ")])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

# The Minion Game

def minion_game(string):
    consonants = 0
    vowels = 0
    for i in range(len(string)):
        if string[i] in "AEIOU":
            vowels += len(string)-i
        else:
            consonants += len(string)-i
    winner = 'Stuart' if consonants > vowels else 'Kevin' if vowels > consonants else None
    print(f"{winner} {max(consonants, vowels)}" if winner else "Draw")

if __name__ == '__main__':
    s = input()
    minion_game(s)

# Merge The Tools!

def merge_the_tools(string, k):
    n = len(string)//k
    for i in range(n):
        t = string[i*k:(i+1)*k]
        u = "".join(t[j] for j in range(k) if t[j] not in t[0:j])
        print(u)
    
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

### Sets

# Introduction to Sets

def average(array):
    array_as_set = set(array)
    return sum(array_as_set)/len(array_as_set)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

# Symmetric Difference

input()
a = set([int(i) for i in input().split()])
input()
b = set([int(i) for i in input().split()])
result = sorted(a.union(b).difference(a.intersection(b)))
print("\n".join(str(i) for i in result))

# Set .add()

s = set()
for i in range(int(input())):
    s.add(input())
print(len(s))

# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
for _ in range(int(input())):
    line = input().strip()
    if line == "pop":
        s.pop()
    else:
        cmd, value = line.split()
        s.discard(int(value)) # Tests do not raise so remove->discard is ok
print(sum(s))

# Set .union() Operation

input()
english = set(input().split())
input()
french = set(input().split())
print(len(english.union(french)))

# Set .intersection() Operation

input()
english = set(input().split())
input()
french = set(input().split())
print(len(english.intersection(french)))

# Set .difference() Operation

input()
english = set(input().split())
input()
french = set(input().split())
print(len(english.difference(french)))

# Set .symmetric_difference() Operation

input()
english = set(input().split())
input()
french = set(input().split())
print(len(english.symmetric_difference(french)))

# Set Mutations

input()
A = set(map(int, input().split()))
N = int(input())
for i in range(N):
    cmd, _ = input().strip().split()
    B = set(map(int, input().split()))
    if cmd == "intersection_update":
        A.intersection_update(B)
    elif cmd == "difference_update":
        A.difference_update(B)
    elif cmd == "update":
        A.update(B)
    else:
        A.symmetric_difference_update(B)
print(sum(A))

# The Captain's Room

rooms_all = set()
rooms_not_captain = set()
input()
for i in map(int, input().strip().split()):
    if i in rooms_all:
        rooms_not_captain.add(i)
    rooms_all.add(i)
print(rooms_all.difference(rooms_not_captain).pop())

# Check Subset

T = int(input())
for t in range(T):
    input()
    A = set(map(int, input().split()))
    input()
    B = set(map(int, input().split()))
    print(len(A.difference(B)) == 0)

# Check Strict Superset

A = set(map(int, input().split()))
N = int(input())
is_superset = True
for i in range(N):
    B = set(map(int, input().split()))
    if not (len(A.difference(B)) > 0 and len(B.difference(A)) == 0):
        is_superset = False
print(is_superset)

# No Idea!

input()
array = list(map(int, input().strip().split()))
A = set(map(int, input().strip().split()))
B = set(map(int, input().strip().split()))
happiness = 0
for i in array:
    if i in A:
        happiness += 1
    elif i in B:
        happiness -= 1
print(happiness)

### Collections

# collections.Counter()

from collections import Counter

input()
available = Counter(map(int, input().strip().split()))
total_gained = 0
for i in range(int(input())):
    size, price = map(int, input().strip().split())
    if available[size] > 0:
        available.subtract(Counter([size]))
        total_gained += price
print(total_gained)

# DefaultDict Tutorial

from collections import defaultdict

A = defaultdict(list)
n,m = map(int, input().split())
for i in range(n):
    A[input()].append(i+1)
for i in range(m):
    c = input()
    if A[c]:
        print(" ".join(map(str, A[c])))
    else:
        print(-1)

# Collections.namedtuple()

n, columns = int(input()), input().split()
marks = [int(input().split()[columns.index("MARKS")]) for i in range(n)]
print(sum(marks) / len(marks))

# Collections.OrderedDict()

from collections import OrderedDict

n = int(input())
d = OrderedDict()
for i in range(n):
    tokens = input().rsplit(" ", 1)
    item, price = tokens[0], int(tokens[1])
    d[item] = d.get(item, 0) + price
for k,v in d.items():
    print(f"{k} {v}")

# Collections.deque()

from collections import deque

n = int(input())
d = deque()
for i in range(n):
    tokens = input().split()
    # Using getattr to save the cumbersome if-elif-...-else chain
    # Either call the method by itself, or also pass its argument
    if len(tokens) == 1:
        getattr(d, tokens[0])()
    else:
        getattr(d, tokens[0])(tokens[1])
print(" ".join(str(i) for i in d))

# Word Order

from collections import OrderedDict
n = int(input())
d = OrderedDict()
for i in range(n):
    line = input()
    d[line] = d.get(line, 0) + 1
print(len(d))
print(" ".join(str(v) for k,v in d.items()))

# Company Logo

from collections import Counter

c = Counter(input())
occurrences = reversed(sorted(set(c.values())))
res = []
"""
Go from highest to lowest occurrences count.
For each group of chars with the same occurrence count, sort them 
alphabetically independently from the rest.
""" 
for occs in occurrences:
    res.extend(sorted([k for k,v in c.items() if v==occs]))
for i in range(3):
    print(res[i], c[res[i]])

# Piling Up!

from collections import deque
T = int(input())
for t in range(T):
    input()
    blocks = deque(map(int, input().split()))
    biggest_block = max(blocks)+1 # For simplicity, start with a block bigger than all the others
    stackable = True
    while blocks:
        if blocks[0] > biggest_block or blocks[-1] > biggest_block:
            stackable = False
            break
        if blocks[0] >= blocks[-1]:
            biggest_block = blocks[0]
            blocks.popleft()
        else:
            biggest_block = blocks[-1]
            blocks.pop()
    print("Yes" if stackable else "No")

### Date and Time

# Calendar Module

from calendar import weekday

m,d,y = map(int, input().split())
w = weekday(y,m,d)
weekdays = ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
print(weekdays[w])

# Time Delta

#!/bin/python3

import math
import os
from datetime import datetime

def time_delta(t1, t2):
    dates = [datetime.strptime(d, "%a %d %b %Y %H:%M:%S %z") for d in [t1,t2]]
    return str(int(abs((dates[0] - dates[1]).total_seconds())))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()
        t2 = input()

        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')

    fptr.close()

### Exceptions

# Exceptions

t = int(input())
for _ in range(t):
    try:
        a,b = map(int, input().split())
        print(a//b)
    except ZeroDivisionError:
        print("Error Code: integer division or modulo by zero")
    except Exception as e:
        print(f"Error Code: {e}")

### Built-ins

# Zipped!

N,X = map(int, input().split())
subject_grades = [map(float,input().split()) for _ in range(X)]
for student_grades in zip(*subject_grades):
    student_avg = sum(student_grades)/X
    print(f"{student_avg:.1f}")

# Athlete Sort

#!/bin/python3

n,m = map(int, input().split())

arr = []
for _ in range(n):
    arr.append(list(map(int, input().rstrip().split())))

k = int(input())
arr.sort(key=lambda v: v[k])
for row in arr:
    print(" ".join(map(str, row)))

# ginortS

import string
s = input()
lower = [c for c in s if c in string.ascii_lowercase]
upper = [c for c in s if c in string.ascii_uppercase]
odds = [c for c in s if c in "13579"]
evens = [c for c in s if c in "02468"]

result = "".join(sorted(lower)+sorted(upper)+sorted(odds)+sorted(evens))
print(result)

### Python Functionals

# Map and Lambda Function

cube = lambda x: x**3

def fibonacci(n):
    if n == 0:
        return []
    if n == 1: 
        return [0]
    if n == 2:
        return [0, 1]
    prev = fibonacci(n-1)
    return prev + [prev[-1]+prev[-2]]

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

### Regex and Parsing challenges

# Detect Floating Point Number

import re

n = int(input())
for _ in range(n):
    print(re.match("^[+-]?[0-9]*\.[0-9]+$", input()) is not None)

# Re.split()

regex_pattern = r"[,.]"

import re
print("\n".join(re.split(regex_pattern, input())))

# Group(), Groups() and Groupdict()

import re

line = input()
search_result = re.search(r"([a-zA-Z0-9])\1", line)
print(search_result.group(1) if search_result else -1)

# Re.findall() & Re.finditer()

import re

line = input()
consonants = "qwrtypsdfghjklzxcvbnm"
consonants += consonants.upper()
vowels = "aeiouAEIOU"
regex = f"(?<=[{consonants}])([{vowels}]{{2,}})(?=[{consonants}])"
result = []
for match in re.finditer(regex, line):
    result.append(match.group())
if result:
    for match in result:
        print(match)
else:
    print(-1)        

# Re.start() & Re.end()

import re
s, k = input(), input()

found = False
i = 0
while True:
    m = re.search(k, s[i:])
    if m:
        print(f"({i+m.start()}, {i+m.end()-1})")
        found = True
        i += m.start()+1
    else:
        break
if not found:
    print("(-1, -1)")

# Regex Substitution

import re  

n = int(input())
for line in [input() for _ in range(n)]:
    line = re.sub("(?<= )&&(?= )", "and", line)
    line = re.sub("(?<= )\|\|(?= )", "or", line) 
    print(line)

# Validating Roman Numerals

regex_pattern = r"^M{0,3}"
regex_pattern += r"(CM|CD|D?C{0,3})"
regex_pattern += r"(XC|XL|L?X{0,3})"
regex_pattern += r"(IX|IV|V?I{0,3})$"

import re
print(str(bool(re.match(regex_pattern, input()))))

# Validating Phone Numbers

import re
n = int(input())
for _ in range(n):
    print("YES" if re.match("^[789][0-9]{9}$", input()) is not None else "NO")

# Validating and Parsing Email Addresses

import email.utils
import re

n = int(input())
for _ in range(n):
    n,e = email.utils.parseaddr(input())
    if re.match("^[a-zA-Z][a-zA-Z0-9\.\-_]+@[a-zA-Z]+\.[a-zA-Z]{1,3}$", e):
        print(f"{n} <{e}>")

# Hex Color Code

import re

num_lines = int(input())
input_oneline = "".join(input() for _ in range(num_lines))

tokens = []
for match in re.findall("\{(.+?)\}", input_oneline):
    for token in re.sub("[;:,\(\)]", " ", match.strip()).split(" "):
        if re.match("#[a-fA-F0-9]{3}", token) or re.match("#[a-fA-F0-9]{6}", token):
            tokens.append(token)
print("\n".join(tokens))

# HTML Parser - Part 1

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")
    def handle_endtag(self, tag):
        print(f"End   : {tag}")
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

parser = MyHTMLParser()
parser.feed("\n".join(input() for _ in range(int(input()))))

# HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        comment_type = "Multi-line" if "\n" in data else "Single-line"
        print(f">>> {comment_type} Comment\n{data}")
    def handle_data(self, data):
        if data.strip():
            print(f">>> Data\n{data}")
  
html = "\n".join(input().rstrip() for _ in range(int(input())))       
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")
    def handle_startendtag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

parser = MyHTMLParser()
parser.feed("\n".join(input() for _ in range(int(input()))))

# Validating UID

import re

for _ in range(int(input())):
    uid = input()
    ok1 = re.match(r"^[a-zA-Z0-9]{10}$", uid) is not None
    ok2 = re.match(r".*[A-Z].*[A-Z].*", uid) is not None
    ok3 = re.match(r".*[0-9].*[0-9].*[0-9].*", uid) is not None
    ok4 = len(set(uid)) == 10
    print("Valid" if all([ok1,ok2,ok3,ok4]) else "Invalid")

# Validating Credit Card Numbers

import re

for i in range(int(input())):
    line = input()

    ok = re.search("^([0-9]{4}(-?)){3}[0-9]{4}$", line) is not None
    line = line.replace("-", "")
    ok &= re.search("^[456]", line) is not None
    ok &= re.search(r"(.)\1{3}", line) is None

    print("Valid" if ok else "Invalid")

# Validating Postal Codes

regex_integer_in_range = r"^[1-9][0-9]{5}$"	
regex_alternating_repetitive_digit_pair = r'([0-9])(?=.\1)'	

import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

# Matrix Script

import re

n,m = map(int, input().split())

lines = [input() for _ in range(n)]
converted = []
for j in range(m):
    for i in range(n):
        converted += lines[i][j]
converted = "".join(converted)

try:
    first_alphanum = re.search("[a-zA-Z0-9]", converted).start()
    last_alphanum = len(converted)-re.search("[a-zA-Z0-9]", converted[::-1]).start()-1

    tokens = [token for token in re.sub("[^a-zA-Z0-9]", " ", converted[first_alphanum:last_alphanum+1]).split(" ")]
    tokens = filter(lambda token: len(token) > 0, tokens)
    print(converted[:first_alphanum] + " ".join(tokens) + converted[last_alphanum+1:])
except:
    print(converted) # To deal with matrices without even a single alphanumeric

### XML

# XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    score = 0
    for n in node.iter():
        score += len(n.attrib)
    return score

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# XML2 - Find the Maximum Depth

import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    d = max(depth(node, level+1) for node in elem)+1 if len(elem)>0 else 0
    maxdepth = max(maxdepth, d)
    return d

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

### Closures and decorations

# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        numbers = []
        for number in l:
            if number[0] == "0":
                numbers.append(number[1:])
            elif number[0:2] == "91" and len(number)>10:
                numbers.append(number[2:])
            elif number[0:3] == "+91":
                numbers.append(number[3:])
            else:
                numbers.append(number)
        numbers = [f"+91 {number[0:5]} {number[5:]}" for number in numbers]
        f(numbers)
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 

# Decorators 2 - Name Directory

import operator

from operator import itemgetter

def person_lister(f):
    def fun(people):
        people.sort(key = lambda x: int(x[2]))
        return [f(p) for p in people]
    return fun

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

### Numpy

# Arrays

import numpy

def arrays(arr):
    return numpy.array(list(reversed([float(i) for i in arr])), float)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

# Shape and Reshape

import numpy

print(numpy.reshape(numpy.array([int(i) for i in input().split()]), (3,3)))

# Transpose and Flatten

import numpy

n,m = map(int, input().split())
arr = []
for i in range(n):
    arr.append([int(i) for i in input().split()])
print(numpy.array(arr).transpose())
print(numpy.array(arr).flatten())

# Concatenate

import numpy

n,m,p = map(int, input().split())
arr = []
for i in range(n):
    arr.append([int(i) for i in input().split()])
A = numpy.array(arr)
arr = []
for i in range(m):
    arr.append([int(i) for i in input().split()])
B = numpy.array(arr)
print(numpy.concatenate((A,B), axis=0))

# Zeros and Ones

import numpy

d = tuple(map(int, input().split()))
print(numpy.zeros(d, dtype=numpy.int))
print(numpy.ones(d, dtype=numpy.int))

# Eye and Identity

import numpy
numpy.set_printoptions(legacy='1.13')

n,m = map(int, input().split())
print(numpy.eye(n,m))

# Array Mathematics

import numpy

n,m = map(int, input().split())
A = numpy.array([[int(i) for i in input().split()] for _ in range(n)])
B = numpy.array([[int(i) for i in input().split()] for _ in range(n)])
print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A%B)
print(A**B)

# Floor, Ceil and Rint

import numpy
numpy.set_printoptions(legacy='1.13')

arr = numpy.array(list(map(float, input().split())))
print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))

# Sum and Prod

import numpy

n,m = map(int, input().split())
arr = numpy.array([[int(i) for i in input().split()] for _ in range(n)])
print(numpy.product(numpy.sum(arr, axis=0)))

# Min and Max

import numpy

n,m = map(int, input().split())
arr = numpy.array([[int(i) for i in input().split()] for _ in range(n)])
print(numpy.max(numpy.min(arr, axis=1)))

# Mean, Var and Std

import numpy

n,m = map(int, input().split())
arr = numpy.array([[int(i) for i in input().split()] for _ in range(n)])
print(numpy.mean(arr, axis=1))
print(numpy.var(arr, axis=0))
print(numpy.around(numpy.std(arr), 11))

# Dot and Cross

import numpy

n = int(input())
A = numpy.array([[int(i) for i in input().split()] for _ in range(n)])
B = numpy.array([[int(i) for i in input().split()] for _ in range(n)])
print(numpy.matmul(A,B))  # Just use the built-in for simplicity

# Inner and Outer

import numpy

A,B = (numpy.array([int(i) for i in input().split()]) for _ in range(2))
print(numpy.inner(A,B))
print(numpy.outer(A,B))

# Polynomials

import numpy

arr = numpy.array([float(i) for i in input().split()])
print(numpy.polyval(arr, float(input())))

# Linear Algebra

import numpy

n = int(input())
arr = numpy.array([[float(i) for i in input().split()] for _ in range(n)])
print(numpy.around(numpy.linalg.det(arr), decimals=2))

### Problem 2

# Birthday Cake Candles

#!/bin/python3

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    max_height = max(candles)
    return sum([1 if elem == max_height else 0 for elem in candles])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

# Number Line Jumps

#!/bin/python3

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if v1 < v2:
        return "NO"
    # Due to constraints, 10k jumps are enough. 
    # Then we just run a full simulation
    for _ in range(10000):
        x1 += v1
        x2 += v2
        if x1 == x2:
            return "YES"
        if x1 > x2:  # Not necessary, small optimization
            return "NO"
    return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    shared = 5
    cumulative = 0
    for i in range(n):
        liked = shared//2
        cumulative += liked
        shared = liked*3
    return cumulative

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())
    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')
    fptr.close()

# Recursive Digit Sum

#!/bin/python3

import math
import os
import random
import re
import sys

def superDigit(n, k):
    digit_sum = sum(int(char) for char in n)*k
    while digit_sum > 10:
        digit_sum = superDigit(str(digit_sum), 1)
    return digit_sum

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)

    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1

#!/bin/python3

import math
import os
import random
import re
import sys

def print_arr(arr):
    print(" ".join(str(i) for i in arr))

def insertionSort1(n, arr):
    last_element = arr[-1]
    last_is_smallest = all(i>last_element for i in arr[:-1])
    for i in reversed(range(1,n)):
        if arr[i-1] > last_element:
            arr[i] = arr[i-1]
            print_arr(arr)
        else:
            arr[i] = last_element
            print_arr(arr)
            break
    if last_is_smallest:
        arr[0] = last_element
        print_arr(arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

# Insertion Sort - Part 2

# #!/bin/python3

import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(1, len(arr)):
        
        for j in reversed(range(1,i+1)):
            if arr[j] >= arr[j-1]:
                break
            arr[j], arr[j-1] = arr[j-1], arr[j]

        print(" ".join(str(n) for n in arr))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
