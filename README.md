# Seidel’s shortest path algorithm implementation for Computational Complexity Theory 2021  course (HSE)

Algorithm implementation is in [seidel_algo.py](seidel_algo.py) file.

It receives an adjacency matrix as input and returns distance matrix as output. For unreachable vertexes the distance equals 0.

File for building automated testing workflow: [python-package.yml](.github/workflows/python-package.yml)

## Algorithm brief description
A - adjacency matrix with size n

Â - adjacency matrix with 1s on diagonal

G - graph

D - distance matrix
1. Find Â^1, Â^2, Â^4... Â^n1 where n1 >= n
2. D of G^n1 = A^n1 (with zeros on diagonal)
3. Find by special rule D of G^n1 -> D of G^n1 / 2 -> ... -> D of G^2 -> D of G
4. Return D of G

## Run
Load dependencies:
```pip install -r requirements.txt```

Run algorithm with example input:
```python seidel_algo.py```

Run tests:
```py.test -v test_seidel.py```