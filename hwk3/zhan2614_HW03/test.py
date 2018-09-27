from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR
from pprint import pprint as pp

And = AND()
And.train()
print("AND Gate test cases")
pp(And.and_nn.theta)
print("and(False, False) = %r" % And(False, False))
print("and(True, False) = %r" % And(True, False))
print("and(False, True) = %r" % And(False, True))
print("and(True, True) = %r \n" % And(True, True))
Or = OR()
Or.train()
pp(Or.or_nn.theta)
print("or(False, False) = %r" % Or(False, False))
print("or(True, False) = %r" % Or(True, False))
print("or(False, True) = %r" % Or(False, True))
print("or(True, True) = %r\n" % Or(True, True))

print("NOT Gate test cases")
Not = NOT()
Not.train()
pp(Not.not_nn.theta)
print("not(False) = %r" % Not(False))
print("not(True) = %r\n" % Not(True))
print("XOR Gate test cases")
Xor = XOR()
Xor.train()
pp(Xor.xor_nn.theta)
print("xor(False, False) = %r" % Xor(False, False))
print("xor(True, False) = %r" % Xor(True, False))
print("xor(False, True) = %r" % Xor(False, True))
print("xor(True, True) = %r" % Xor(True, True))
print("OR Gate test cases")

