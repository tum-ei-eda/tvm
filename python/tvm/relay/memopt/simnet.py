from . import bufferopt
import numpy as np
import networkx as nx
from collections import defaultdict, deque


def doesOverlap(start1, size1, start2, size2):
    return (start1 < (start2+size2)) and (start2 < (start1+size1))


# Represents a buffer that is used in LayerOps.
class Buffer:
    def __init__(self, name, size, static=False):
        self.name = name
        self.size = size
        self.static = static
        # add first/last use?

        if static:
            self.simBuf = np.linspace(0.0, 100.0, size)
        else:
            self.simBuf = np.zeros(size)

    def isStatic(self):
        return self.static

    def plot(self):
        return self.name + "[label=\"" + self.name + " (" + str(self.size) + ")\", style=filled, color=" + ("gray40" if self.static else "gray80") + "];\n"

    def getSimBuf(self):
        return self.simBuf

    def setSimBuf(self, arr):
        assert(arr.shape == self.simBuf.shape)
        self.simBuf = arr

    def dbgVal(self):
        return np.sum(self.simBuf)

    def __repr__(self):
        return f"Buffer('{self.name}', {self.size}, {self.static})={self.dbgVal()}"


# Represents an abstract operation that takes inputs and produces outputs.
class LayerOp:
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

    def getInputs(self):
        return self.inputs

    def getOutputs(self):
        return self.outputs

    def getInSize(self):
        return sum([buf.size for buf in self.inputs if not buf.isStatic()])

    def getOutSize(self):
        return sum([buf.size for buf in self.outputs])

    # Returns the required memory size for non-static inputs and outputs.
    def getSize(self):
        return self.getInSize() + self.getOutSize()

    def plot(self):
        out = self.name + "[shape=box, style=filled, color=coral];\n"
        for i in self.inputs:
            out += i.plot() + i.name + " -> " + self.name + ";\n"
        for i in self.outputs:
            out += i.plot() + self.name + " -> " + i.name + ";\n"
        return out

    def sim(self):
        for out in self.outputs:
            outBuf = out.getSimBuf()
            outShape = outBuf.shape

            for i in self.inputs:
                tempB = np.resize(i.getSimBuf(), outShape)
                outBuf = np.add(outBuf, tempB)
                out.setSimBuf(outBuf)

    def __repr__(self):
        return f"LayerOp('{self.name}')[{len(self.inputs)} -> {len(self.outputs)}]"


# Represents a collection of Buffers and LayerOps to form a data flow graph.
class Network:
    def __init__(self):
        self.bufs = []
        self.ops = []

    def addBuf(self, *args):
        buf = Buffer(*args)
        self.bufs.append(buf)
        return buf

    def addOp(self, *args):
        op = LayerOp(*args)
        self.ops.append(op)
        return op

    def loadFromDeepThings(self, m):
        inBuf = self.addBuf("B0", m.S[0])
        for i in range(0, len(m.CONV)):
            wBuf = self.addBuf(f"B{i}w", m.W[i], True)
            outBuf = self.addBuf(f"B{i+1}", m.S[i+1])
            self.addOp(f"L{i}", [inBuf, wBuf], [outBuf])
            inBuf = outBuf

        self.createGraph()

    def loadFromTFLite(self, m):
        pass

    def splitLargest(self):
        maxSize = 0
        largestOp = None
        for op in self.ops:
            if op.getSize() > maxSize:
                maxSize = op.getSize()
                largestOp = op

        if not largestOp:
            return

        # TODO: the largest buf must have preds and succs to be egligable for splitting
        # so we do not need both code paths below.

        if largestOp.getInSize() >= largestOp.getOutSize():
            if len(self.g.in_edges(largestOp)) == 0:
                print("Cannot split largest because input has no predecessors")
                return
            inputsForLargest = []
            for pred in self.g.predecessors(largestOp):
                print("PRED:", pred)
                splitBufs = []
                bufsToKeepForLargest = []
                bufsToKeepForLargest.extend(largestOp.getInputs())
                for outBuf in pred.getOutputs():
                    if outBuf in largestOp.getInputs():
                        bufsToKeepForLargest.remove(outBuf)
                        splitBufs.append(self.addBuf(outBuf.name + "b", outBuf.size // 2, outBuf.isStatic()))
                        outBuf.name += "a"
                        outBuf.size = outBuf.size // 2
                    # else: handling of other outs would be op specific.

                self.addOp(pred.name + "b", pred.inputs, splitBufs)
                pred.name += "a"
                inputsForLargest.extend(splitBufs)
                inputsForLargest.extend(bufsToKeepForLargest)

            self.addOp(largestOp.name + "b", inputsForLargest, largestOp.outputs)
            largestOp.name += "a"

        else:
            if len(self.g.out_edges(largestOp)) == 0:
                print("Cannot split largest because output has no successors")
                return
            outputsForLargest = []
            for succ in self.g.successors(largestOp):
                print("SUCC:", succ)
                splitBufs = []
                bufsToKeepForLargest = []
                bufsToKeepForLargest.extend(largestOp.getOutputs())
                for inBuf in succ.getInputs():
                    if inBuf in largestOp.getOutputs():
                        bufsToKeepForLargest.remove(inBuf)
                        splitBufs.append(self.addBuf(inBuf.name, + "b", inBuf.size // 2, inBuf.isStatic()))
                        inBuf.name += "a"
                        inBuf.size = inBuf.size // 2

                self.addOp(succ.name + "b", succ.outputs, splitBufs)
                succ.name += "a"
                outputsForLargest.extend(splitBufs)
                outputsForLargest.extend(bufsToKeepForLargest)

            self.addOp(largestOp.name + "b", outputsForLargest, largestOp.inputs)
            largestOp.name += "a"

        self.createGraph()

    def createGraph(self):
        self.g = nx.DiGraph()

        for op in self.ops:
            for outBuf in op.getOutputs():
                for opTarget in self.ops:
                    if outBuf in opTarget.getInputs():
                        self.g.add_edge(op, opTarget)

    def createSchedules(self):
        scheds = []
        for ops in nx.all_topological_sorts(self.g):
            sched = Schedule()
            for op in ops:
                sched.addOp(op)
            scheds.append(sched)

        return scheds

    def plot(self):
        out = ""
        for op in self.ops:
            out += op.plot()
        return out


class Lifetime:
    def __init__(self):
        self.firstUse = 0xffffffff
        self.lastUse = -1

    def addUse(self, t):
        self.firstUse = min(t, self.firstUse)
        self.lastUse = max(t, self.lastUse)


class Lifetimes:
    def __init__(self):
        self.ranges = defaultdict(Lifetime)
        self.ts = defaultdict(set)

    def addUse(self, t, key):
        lt = self.ranges[key]
        lt.addUse(t)
        for i in range(lt.firstUse, lt.lastUse + 1):
            self.ts[i].add(key)

    def getFirstUse(self, key):
        return self.ranges[key].firstUse

    def getLastUse(self, key):
        return self.ranges[key].lastUse

    def getKeysAt(self, t):
        return self.ts[t]


class MemoryLayout:
    def __init__(self):
        self.numBuckets = 0
        self.buckets = []
        self.bufToOffset = {}
        self.bufToBucket = {}
        self.bucketSizes = {}

    def newBucket(self):
        self.buckets.append([])
        self.bucketSizes[self.numBuckets] = 0
        self.numBuckets += 1
        return self.numBuckets - 1

    def fitsIntoBucket(self, buf, offset, index):
        for otherBuf in self.buckets[index]:
            if doesOverlap(offset, buf.size, self.bufToOffset[otherBuf], otherBuf.size):
                return False

        return True

    def addBufToBucket(self, buf, index, offset=None):
        if offset != None:
            if not self.fitsIntoBucket(buf, offset, index):
                raise RuntimeError("WRONG BUFFER ASSIGNMENT")
        else:
            offset = self.bucketSizes[index]

        self.buckets[index].append(buf)
        self.bufToOffset[buf] = offset
        self.bufToBucket[buf] = index
        self.bucketSizes[index] = max(self.bucketSizes[index], buf.size + offset)
        #print("adding " + buf.name + " to bucket " + str(index) + " with offset " + str(offset))

    def addBufToOffset(self, buf, offset):
        # Find free bucket.
        freeBucket = None
        for i in range(0, len(self.buckets)):
            if self.fitsIntoBucket(buf, offset, i):
                freeBucket = i
                break

        if freeBucket == None:
            freeBucket = self.newBucket()

        self.addBufToBucket(buf, freeBucket, offset)

    def getOffset(self, buf):
        return self.bufToOffset[buf]

    def getBucket(self, buf):
        return self.bufToBucket[buf]

    def isPlaced(self, buf):
        return buf in self.bufToOffset

    def getSize(self):
        maxSize = 0
        for i, sz in self.bucketSizes.items():
            maxSize = max(maxSize, sz)
        return maxSize

    def __repr__(self):
        out = "MemLayout():"
        for i, b in enumerate(self.buckets):
            out += "- Bucket " + str(i) + "\n"
            for buf in b:
                out += "  - Buf: " + str(buf) + "\n"
        out += "----- sz: " + str(self.getSize())
        return out


class MemoryPlanner:
    def __init__(self, sched):
        self.sched = sched

        self.conflictGraph = nx.Graph()
        self.lifetimes = Lifetimes()

        for i, op in enumerate(self.sched.sched):
            for inBuf in op.getInputs():
                if not inBuf.isStatic():
                    self.lifetimes.addUse(i, inBuf)
            for outBuf in op.getOutputs():
                self.lifetimes.addUse(i, outBuf)

        for t in range(0, len(self.sched.sched)):
            bufsAtT = list(self.lifetimes.getKeysAt(t))
            for i in range(0, len(bufsAtT)):
                for j in range(i + 1, len(bufsAtT)):
                    self.conflictGraph.add_edge(bufsAtT[i], bufsAtT[j])

    def createOptimalLayout(self):
        minSize = 0xffffffff

        bufs = {}
        bufSizes = []
        for i, buf in enumerate(self.conflictGraph.nodes()):
            bufs[buf] = i
            bufSizes.append(buf.size)

        conflicts = []
        for buf, i in bufs.items():
            otherBufs = list(self.conflictGraph.neighbors(buf))
            for otherBuf in otherBufs:
                j = bufs[otherBuf]
                if i < j:
                    conflicts.append((i, j))

        opt = bufferopt.BufferOpt()
        offsets = opt.solve(bufSizes, conflicts)

        memLayout = MemoryLayout()
        for buf, i in bufs.items():
            memLayout.addBufToOffset(buf, offsets[i])
        return memLayout

    def createColoringLayout(self):
        memLayout = MemoryLayout()

        colorToSlot = {}
        slotToBucketsUsed = {}
        bufToSlot = {}
        slotToSize = {}

        for buf, c in nx.coloring.greedy_color(self.conflictGraph).items():
            if c not in colorToSlot:
                colorToSlot[c] = len(colorToSlot)
                slotToBucketsUsed[colorToSlot[c]] = 0

            slot = colorToSlot[c]
            bucket = slotToBucketsUsed[slot]
            if bucket >= memLayout.numBuckets:
                newBucket = memLayout.newBucket()
                assert newBucket == bucket

            slotToBucketsUsed[slot] += 1

            bufToSlot[buf] = slot
            if slot not in slotToSize:
                slotToSize[slot] = buf.size
            else:
                slotToSize[slot] = max(slotToSize[slot], buf.size)

        for buf, slot in bufToSlot.items():
            offset = 0
            for i in range(0, slot):
                offset += slotToSize[i]
            memLayout.addBufToOffset(buf, offset)

        return memLayout

    def createTFLMLayout(self):
        memLayout = MemoryLayout()

        # Sort buffers by size descending.
        sortedBufs = sorted(self.conflictGraph.nodes(), key=lambda x: x.size, reverse=True)

        # Place first buffer.
        memLayout.newBucket()
        memLayout.addBufToBucket(sortedBufs[0], 0)

        # Loop through in descending order.
        for i in range(1, len(sortedBufs)):
            buf = sortedBufs[i]

            # Look for other buffers that need to be in memory at the same time.
            otherBufs = list(self.conflictGraph.neighbors(buf))

            # Use the first available gap.
            otherBufPlacements = []
            otherPlacedBufs = [b for b in otherBufs if memLayout.isPlaced(b)]
            otherPlacedBufs.sort(key=lambda x: memLayout.getOffset(x))
            currentOffset = 0
            while True:
                gapFound = True
                for otherBuf in otherPlacedBufs:
                    otherOffset = memLayout.getOffset(otherBuf)
                    if doesOverlap(currentOffset, buf.size, otherOffset, otherBuf.size):
                        currentOffset = max(currentOffset, otherOffset + otherBuf.size)
                        gapFound = False
                        break
                if gapFound:
                    break

            memLayout.addBufToOffset(buf, currentOffset)

        return memLayout


# Represents a linear order of LayerOps.
class Schedule:
    def __init__(self):
        self.sched = []

    def addOp(self, op):
        self.sched.append(op)

    def plot(self):
        out = ""
        prevOp = None
        for op in self.sched:
            out += op.plot()
            if prevOp:
                out += prevOp.name + " -> " + op.name + " [color=red];\n"
            prevOp = op
        return out

    def __repr__(self):
        return "[" + "] > [".join([op.name for op in self.sched]) + "]"


# Represents a
class SimNet:
    pass
