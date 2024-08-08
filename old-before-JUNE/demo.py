import ast
import codegen

expr="""
def foo():
   print("hello world")
"""
expr2="""
sum = 0
for i in 5:
  if i%2 == 0:
    if i%3 == 0:
      sum += i
    else:
      sum += 1
  else:
    sum -= i
"""
p=ast.parse(expr2)
print(ast.dump(p.body[0]))
print(ast.dump(p.body[1]))
#p.body[0].body = [ ast.parse("return 0").body[0] ] # Replace function body with "return 0"
import sys
#print(sys.path)
print(codegen.to_source(p))
