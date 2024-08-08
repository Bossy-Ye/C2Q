import ast
import mycodegen

expr="""
def euclidean_division(dividend,divisor):
  if divisor == 0:
    return "divisor cannot be 0"
  a = abs(dividend)
  b = abs(divisor)
  q = 0
  r = 0
  while a>=b:
    a -= b
    q += 1
  flag = dividend * divisor
  if flag > 0:
    sign = 1
  else:
    sign = -1
  
  return (sign*q,r)

dividend = 20
divisor = 3
quotient, remainder = euclidean_division(dividend, divisor)
"""
p=ast.parse(expr)
print(mycodegen.to_source(p))
