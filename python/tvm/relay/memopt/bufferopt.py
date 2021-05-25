from ortools.linear_solver import pywraplp
from . import ilpsolver


class BufferOpt:
    def __init__(self):
        self.ilp = ilpsolver.ILPSolver()
        self.s = self.ilp.s

    def solve(self, sz, conflicts):
        numBufs = len(sz)

        o = []
        for i in range(0, numBufs):
            o.append(self.s.IntVar(0, self.s.Infinity(), "o_" + str(i)))

        # Buffer conflicts.
        for i, j in conflicts:
            print("adding conflict constraint", i, sz[i], j, sz[j])
            self.makeNoOverlap(o[i], sz[i], o[j], sz[j])

        # endOffset_i = o_i + sz_i
        endOffsets = []
        for i in range(0, numBufs):
            endOffsets.append(self.s.IntVar(0, self.s.Infinity(), "endOffset_" + str(i)))
            ct = self.s.Constraint(sz[i], sz[i], "ct_endoffset_"+ str(i))
            ct.SetCoefficient(endOffsets[i], 1)
            ct.SetCoefficient(o[i], -1)

        # max(endOffsets)
        obj = self.ilp.getObjectiveFunc()
        obj.SetCoefficient(self.ilp.makeMaxVar(endOffsets, "totalSz"), 1)

        self.ilp.solve()

        print("totalSz:", obj.Value())

        outOffsets = []
        for i in range(0, numBufs):
            outOffsets.append(o[i].solution_value())

        return outOffsets

    def makeNoOverlap(self, o_i, sz_i, o_j, sz_j):
        # overlap if:
        # o_i < o_j + sz_j  &&  o_j < o_i + sz_i
        # therefor no overlap if:
        # o_i >= o_j + sz_j  ||  o_j >= o_i + sz_i
        # rewritten as less equal:
        # o_j - o_i <= -sz_j
        # o_i - o_j <= -sz_i

        ct1, ct2 = self.ilp.makeOrConstraint(-sz_j, -sz_i)
        ct1.SetCoefficient(o_j, 1)
        ct1.SetCoefficient(o_i, -1)
        ct2.SetCoefficient(o_i, 1)
        ct2.SetCoefficient(o_j, -1)

