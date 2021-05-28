from collections import defaultdict
import numpy as np
import tvm._ffi
from .memopt import simnet as sn
from .memopt import drawing
from tvm import relay



import subprocess
import shutil
def writePlot(name, plotStr):
    fullStr = "digraph G {\n" + plotStr + "}\n"
    with open("/tmp/g.dot", "w") as f:
        f.write(fullStr)
    subprocess.run(["dot", "-O", "-Tpng", "/tmp/g.dot"], check=True)
    shutil.move("/tmp/g.dot.png", "./" + name + ".png")
    import os
    print("CWD:", os.getcwd())



class BufferInfo:
    def __init__(self, device_type):
        self.device_type = device_type
        self.size = 0
        self.firstUse = -1
        self.lastUse = -1
        self.static = False


def divRoundUp(sz, word_sz):
    return (sz + word_sz - 1) // word_sz

def getTypeSize(t):
    assert isinstance(t, relay.ty.TensorType)
    size = 1
    for dim in t.shape:
        size *= dim
    ty2Sz = {
        "int8": 8,
        "int16": 16,
        "int32": 32,
        "int64": 64,
        "float32": 32
    }
    size *= divRoundUp(ty2Sz[t.dtype], 8)
    return int(size)


def get_post_dfs_order_exprs(expr):
    out = []
    def visit(node):
        out.append(node)
    relay.analysis.post_order_visit(expr, visit)
    return out


class CallAnalyzer(relay.expr_functor.ExprVisitor):
    def __init__(self):
        super().__init__()

        self.exprToBufInfos = {}
        self.exprToLabel = {}

    def run(self, func):
        self.nodeDeviceMap = relay.analysis.collect_device_info(func)
        self.orderedExprs = get_post_dfs_order_exprs(func)

        for param in func.params:
            self.updateOrCreateBufInfos(param, False, param)

        self.visit(func.body)

        # Verify consistency.
        for bufInfos in self.exprToBufInfos.values():
            firstUse = -2
            lastUse = -2
            for bufInfo in bufInfos:
                if firstUse == -2:
                    firstUse = bufInfo.firstUse
                    lastUse = bufInfo.lastUse
                else:
                    assert firstUse == bufInfo.firstUse
                    assert lastUse == bufInfo.lastUse

        callLabelId = 0
        varLabelId = 0
        for expr in self.orderedExprs:
            if expr in self.exprToBufInfos:
                if isinstance(expr, relay.Call):
                    self.exprToLabel[expr] = "C" + str(callLabelId)
                    callLabelId += 1
                elif isinstance(expr, relay.Var):
                    self.exprToLabel[expr] = "V" + str(varLabelId)
                    varLabelId += 1

        for expr, bufInfos in self.exprToBufInfos.items():
            if len(bufInfos) != 1:
                print("unexpected bufs len:", len(bufInfos))
            if expr in self.exprToLabel:
                print(self.exprToLabel[expr], bufInfos[0].firstUse, bufInfos[0].lastUse)


    def updateOrCreateBufInfos(self, op, inputToNode, useNode, static=False):
        if op not in self.exprToBufInfos:
            self.createBufInfos(op, static)
        bufInfos = self.exprToBufInfos[op]

        t = self.orderedExprs.index(useNode)
        for bufInfo in bufInfos:
            if inputToNode:
                bufInfo.lastUse = max(bufInfo.lastUse, t)
            else:
                if bufInfo.firstUse == -1:
                    bufInfo.firstUse = t
                else:
                    bufInfo.firstUse = min(bufInfo.firstUse, t)

    def createBufInfos(self, op, static):
        assert op not in self.exprToBufInfos
        bufInfos = []
        device_type = self.nodeDeviceMap.get(op, 0)

        def makeBuf(t):
            bufInfo = BufferInfo(device_type)
            bufInfo.size = getTypeSize(t)
            bufInfo.static = static
            return bufInfo

        if isinstance(op._checked_type_, relay.ty.TupleType):
            for t in op._checked_type_.fields:
                bufInfos.append(makeBuf(t))
        else:
            bufInfos.append(makeBuf(op._checked_type_))

        self.exprToBufInfos[op] = bufInfos

    def visit_function(self, func):
        # Do not recurse into sub functions.
        pass

    def visit_constant(self, const):
        self.updateOrCreateBufInfos(const, False, const, True)

    def visit_call(self, call):
        # Buffer used as output.
        self.updateOrCreateBufInfos(call, False, call)

        # Buffers used as input.
        for arg in call.args:
            self.visit(arg)
            self.updateOrCreateBufInfos(arg, True, call)

    def makeNet(self):
        n = sn.Network()

        exprToBufs = defaultdict(list)
        self.bufToExpr = {}
        self.bufToBufInfo = {}
        for expr, bufInfos in self.exprToBufInfos.items():
            for bufNum, bufInfo in enumerate(bufInfos):
                if expr in self.exprToLabel:
                    labelExt = "_out" + (bufNum if bufNum != 0 else "")
                    buf = n.addBuf(self.exprToLabel[expr] + labelExt, bufInfo.size, bufInfo.static)
                    exprToBufs[expr].append(buf)
                    self.bufToExpr[buf] = expr
                    self.bufToBufInfo[buf] = bufInfo

        self.exprToOp = {}
        for expr, bufsInfo in self.exprToBufInfos.items():
            if isinstance(expr, relay.Call):
                inBufs = []
                for arg in expr.args:
                    argLabel = "UNK_" + str(type(arg))
                    if arg in self.exprToLabel:
                        argLabel = self.exprToLabel[arg]
                    print("Arg of", self.exprToLabel[expr], ":", argLabel)
                    inBufs.extend(exprToBufs[arg])
                op = n.addOp(self.exprToLabel[expr], inBufs, exprToBufs[expr])
                self.exprToOp[expr] = op

        n.createGraph()
        writePlot("net", n.plot())
        return n

    def getSched(self):
        sched = sn.Schedule()
        for expr in self.orderedExprs:
            if expr in self.exprToOp:
                sched.addOp(self.exprToOp[expr])
                print(self.exprToLabel[expr])
                print(expr)
        return sched

    def getExprFromBuf(self, buf):
        return self.bufToExpr[buf]

    def getBufInfoFromBuf(self, buf):
        return self.bufToBufInfo[buf]

    def getStaticBufInfos(self):
        out = []
        for expr, bufInfos in self.exprToBufInfos.items():
            for bufInfo in bufInfos:
                if bufInfo.static:
                    out.append((expr, bufInfo))
        return out


def drawLayout(name, layout):
    IMGWID = 800
    IMGHEI = 600
    BORDER = 50
    d = drawing.Drawing(IMGWID, IMGHEI, origin=(-BORDER, -BORDER))
    totalSz = layout.getSize()
    totalWid = IMGWID - 2*BORDER
    addrScale = totalWid / totalSz
    for i, b in enumerate(layout.buckets):
        for buf in b:
            offset = layout.getOffset(buf)
            x = offset * addrScale
            wid = buf.size * addrScale
            y = i * 35
            hei = 30
            d.textBox(x, y, wid, hei, buf.name)

    d.save(name + ".svg")


@tvm._ffi.register_func("tvm.relay.plan_memory")
def _plan_memory(func):
    analyzer = CallAnalyzer()
    analyzer.run(func)

    n = analyzer.makeNet()
    planner = sn.MemoryPlanner(analyzer.getSched())
    memLayout = planner.createOptimalLayout()
    print("Layout size:", memLayout.getSize())
    drawLayout("layout", memLayout)

    # Return Map<Expr, Array<IntegerArray>> where the key is a list of device_id, storage_id, offset
    out = defaultdict(lambda: [[], [], [], []])
    def placeBuf(expr, bufInfo, sid, offset):
        out[expr][0].append(sid)
        out[expr][1].append(bufInfo.device_type)
        out[expr][2].append(bufInfo.size)
        out[expr][3].append(offset)

    for bucket in memLayout.buckets:
        for buf in bucket:
            offset = memLayout.getOffset(buf)
            placeBuf(analyzer.getExprFromBuf(buf), analyzer.getBufInfoFromBuf(buf), 0, int(offset))

    # Add the static bufs.
    storageId = 1
    for expr, bufInfo in analyzer.getStaticBufInfos():
        placeBuf(expr, bufInfo, storageId, 0)
        storageId += 1

    return out
