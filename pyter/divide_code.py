patch = """
--- a/sympy/core/power.py
+++ b/sympy/core/power.py
@@ -268,7 +268,7 @@ class Pow(Expr):
                 return S.One
 
             if e is S.NegativeInfinity:
-                return S.Zero
+                return S.ComplexInfinity
 
             # In this case, limit(b,x,0) is equal to 1.
             if base.is_positive or base.is_negative:
"""

lines = patch.split('\n')

old_solution = []
new_solution = []

for line in lines:
    if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
        continue
    if line.startswith('-'):
        old_solution.append(line[1:])
    if line.startswith('+'):
        new_solution.append(line[1:])

old_solution = '\n'.join(old_solution)
new_solution = '\n'.join(new_solution)

print("Old Solution:\n", old_solution)
print("New Solution:\n", new_solution)