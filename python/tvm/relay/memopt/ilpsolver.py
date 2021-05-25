from ortools.linear_solver import pywraplp

# Tricks are from:
# https://download.aimms.com/aimms/download/manuals/AIMMS3OM_IntegerProgrammingTricks.pdf


class ILPSolver:
    def __init__(self):
        solveType = pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        self.s = pywraplp.Solver("solver", solveType)
        self.genId = 0
        self.M = 1e7

    def getObjectiveFunc(self):
        obj = self.s.Objective()
        obj.SetMinimization()
        return obj

    def solve(self):
        self.s.Solve()

    def nextId(self, name):
        self.genId += 1
        return name + "_" + str(self.genId)

    # Constrains variable v to be equal to value.
    def forceTo(self, v, value):
        ct = self.s.Constraint(value, value, self.nextId("ct_forceto"))
        ct.SetCoefficient(v, 1)

    # Constrains x1-x2 to be within low to high (inclusive).
    def subLimits(self, x1, x2, low, high):
        ct = self.s.Constraint(low, high, self.nextId("ct_sub"))
        ct.SetCoefficient(x1, 1)
        ct.SetCoefficient(x2, -1)

    def lessEqual(self, x1, x2):
        self.subLimits(x2, x1, 0, self.s.Infinity())

    def lessThan(self, x1, x2):
        self.subLimits(x2, x1, 1, self.s.Infinity())

    def lessEqualBool(self, x1, x2):
        # x1 <= x2  <=>  0 <= x2 - x1 <= 1
        self.subLimits(x2, x1, 0, 1)

    def lessThanBool(self, x1, x2):
        # x1 < x2  <=>  1 <= x2 - x1 <= 2
        self.subLimits(x2, x1, 1, 2)

    # Create a BoolVar that is constrained to be equal to x1*x2 on bools.
    def makeMultVar(self, x1, x2, name):
        y = self.s.BoolVar(self.nextId(name))

        self.lessEqualBool(y, x1)
        self.lessEqualBool(y, x2)

        # y >= x1 + x2 - 1  <=>  -1 <= y - x1 - x2 <= 1
        ct3 = self.s.Constraint(-1, 1, self.nextId("ct_mul"))
        ct3.SetCoefficient(y, 1)
        ct3.SetCoefficient(x1, -1)
        ct3.SetCoefficient(x2, -1)

        return y

    def makeMaxVar(self, vars, name):
        # C = max(v_0, ..., v_N)
        C = self.s.IntVar(0, self.s.Infinity(), self.nextId(name))

        h = []
        for i, v in enumerate(vars):
            h.append(self.s.BoolVar(self.nextId("var_max_" + str(i))))

            # C >= v_i
            self.lessEqual(v, C)

            # C <= v_i + (1-h_i)M
            ct = self.s.Constraint(-self.M, self.s.Infinity(), self.nextId("ct_max_" + str(i)))
            ct.SetCoefficient(v, 1)
            ct.SetCoefficient(h[i], -self.M)
            ct.SetCoefficient(C, -1)

        # sum(h_i) = 1
        ctsum = self.s.Constraint(1, 1, self.nextId("ct_max"))
        for hi in h:
            ctsum.SetCoefficient(hi, 1)

        return C

    # Return two constraints where only at least one must hold.
    # They will be lessEqual with the given upper limits.
    def makeOrConstraint(self, upper1, upper2):
        # x <= upper1 or y <= upper2

        h = self.s.BoolVar(self.nextId("var_or"))

        # x - M*h     <= upper1
        ct1 = self.s.Constraint(-self.s.Infinity(), upper1, self.nextId("ct_or1"))
        ct1.SetCoefficient(h, -self.M)

        # y - M*(1-h) <= upper2
        # y + M*h     <= upper2 + M
        ct2 = self.s.Constraint(-self.s.Infinity(), upper2 + self.M, self.nextId("ct_or2"))
        ct2.SetCoefficient(h, self.M)

        return ct1, ct2
