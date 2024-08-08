import ast
import mycodegen

expr="""
def foo():
   print("hello world")
"""
expr2="""
def foo(x,k):
  return 5 * x + k
x, sum = 1.12, 1.13
x = x+0.01
x = x << 1
k=3.333
lists = [1,2,3]
flag=(1==1)
s = 1
y = 1.33
for i in 5:
  if i%2 == 0:
    if i%3 == 0:
      sum += i
      sum = sum+i
    elif i%3 == 1:
      sum = sum+233
    else:
      sum += 1
  else:
    sum -= i
i = 1
while i < 6:
  i += 1
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
