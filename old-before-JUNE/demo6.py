import ast
import mycodegen

expr="""
def foo():
   print("hello world")
"""
expr2="""
def Max(a,b):
  if a>=b:
    return a
  else:
    return b

lists = [2,3]
max = 0
for i in 5:
  if i%2 == 0:
    if i%3 == 0:
      max = 11
      #print('max')
    elif i%3 == 1:
      max += 0
    else:
      max += 1
  else:
    max = max - 1
i = 0
while i < 6:
  i+=1
  if i==4:
    break
  continue
"""
p=ast.parse(expr2)
print(ast.dump(p.body[0]))
print(ast.dump(p.body[1]))
print(ast.dump(p.body[2]))
#p.body[0].body = [ ast.parse("return 0").body[0] ] # Replace function body with "return 0"
#print(sys.path)
print(mycodegen.to_source(p))
