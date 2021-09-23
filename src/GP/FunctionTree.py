''' This module holds object definitions for function trees, plus a tree building function. '''
import numpy as np
import pandas as pd
import datetime as dt
import ipdb
import time
import copy
import sys

sys.path.append('../')
from util.Utils import *


''' a node encapsulates an operation / fatures / constant; it remembers it's parent, and knows it's children '''
class Node(object):
    def __init__(self, typ, val, par):
        '''
        Node constructor. Children nodes are set separately.
        :param typ: string type of Node ('op', 'const', 'feat')
        :param val: value of Node, dependent on type
        :param par: Node object parent Node
        '''
        self.type = typ
        self.value = val
        self.parent = par
        # init the string and left & right to nothing
        self.Str = '%s(%s)'%(self.type, self.value)
        self.left = None
        self.right = None
        self.setStr('LR')
        
    def setNode(self, typ=None, val=None):
        '''
        Function to set / reset this Node's type and/or value.
        :param typ: optional string type of Node ('op', 'const', 'feat')
        :param val: optional value of Node, dependent on type
        '''
        if typ is not None:
            self.type = typ
        if val is not None:
            self.value = val
        self.Str = '%s(%s)'%(self.type, self.value)
        
    def __str__(self):
        return '(%s) -> [%s, %s]'%(self.Str, self.leftStr, self.rightStr)
    
    def setLeft(self, L):
        '''
        Set this Node's left child Node.
        :param L: Node object left child Node
        '''
        self.left = L
        self.setStr('L')
        
    def setRight(self, R):
        '''
        Set this Node's right child Node.
        :param R: Node object right child Node
        '''
        self.right = R
        self.setStr('R')
    
    def setStr(self, LR):
        '''
        Set this string for either or both of this Node's children.
        :param LR: string indicating which child Node's string to
        set ('L', 'R', 'LR')
        '''
        if 'L' in LR:
            if self.left == None:
                self.leftStr = '_'
            else:
                self.leftStr = '%s(%s)'%(self.left.type, self.left.value)
        if 'R' in LR:
            if self.right == None:
                self.rightStr = '_'
            else:
                self.rightStr = '%s(%s)'%(self.right.type, self.right.value)


''' a tree represents the entire function; it knows the root, level-ordered structure, location of leaves, and depth '''
class Tree(object):
    '''
    Tree constructor.
    :param root: Node object for this Tree's root
    :param maxDepth: maximum allowable depth for this Tree, just used for
    initialization
    '''
    def __init__(self, root, maxDepth):
        self.root = root
        self.depth = maxDepth # init actual depth with just the max depth allowed for now
        self.leaves = None
        self.struct = None
        self.function = None
        # build the structure dict & function string
        self.GenStruct()
        
    def __str__(self):
        # first build the struct dict, if necessary
        if self.struct is None:
            self.GenStruct()
        # now print
        return '\n'.join(['%d: %s'%(key, '|'.join([str(node.value) for node in val])) for (key, val) in self.struct.items()])
    
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def __RecTreeStruct(currNode, tree, leaves, currKey):
        '''
        Recursive tree structuring function; only to be called by TreeStruct
        '''
        # save the node
        this = tree[currKey].copy()
        this.append(currNode)
        tree[currKey] = this
        
        # save the node if it's a leaf
        if currNode.type in ['const', 'feat']:
            this = leaves[currKey].copy()
            this.append(currNode)
            leaves[currKey] = this
            
        if (currNode.left is None) & (currNode.right is None):
            return tree, leaves

        if currNode.left is not None:
            tree, leaves = Tree.__RecTreeStruct(currNode.left, tree, leaves, currKey+1)
        if currNode.right is not None:
            tree, leaves = Tree.__RecTreeStruct(currNode.right, tree, leaves, currKey+1)
        return tree, leaves

    def GenStruct(self):
        '''
        Return the function tree structure as a dictionary.
        :return tree: level number-keyed dict of the tree
        '''
        # populate the tree view dict; have to init the dict to a large number, because
        # with crossovers, trees could get very large
        self.struct = dict.fromkeys(range(1000), [])
        self.leaves = dict.fromkeys(range(1000), [])
        self.struct, self.leaves = Tree.__RecTreeStruct(self.root, self.struct, self.leaves, 0)
        # prune now (remove unused rows)
        for key in list(self.struct.keys()):
            if self.struct[key] == []:
                self.struct.pop(key)
        for key in list(self.leaves.keys()):
            if self.leaves[key] == []:
                self.leaves.pop(key)
        
        # set the depth
        self.depth = max(self.struct.keys())+1
        
        # now generate the function
        self.function = self.GenFunction()
        
        return None

    def GenFunction(self):
        '''
        Returns a string representation of the function tree as
        a function. This can also act as the hash of a tree.
        :return function: the string of the function
        '''
        
        funcStrings = {}

        # special handling of const or feat root nodes
        if self.root.type != 'op':
            funcStrings[self.root] = str(self.root.value)
        else:
            # start at the top & climb down the tree
            for currLev in range(self.depth-1, 0, -1):
                nodes = self.struct[currLev]
                # parse the nodes at this level and iterate in pairs
                for indx in range(0, len(nodes), 2):
                    # if there's a func string already defined, use them
                    lVal = funcStrings.get(nodes[indx], str(nodes[indx].value))
                    rVal = funcStrings.get(nodes[indx+1], str(nodes[indx+1].value))
                    # build and store the function string
                    funcStrings[nodes[indx].parent] = nodes[indx].parent.value + '(' + lVal + ',' + rVal + ')'
        
        # get the function & and add 'np.' where necessary
        func = funcStrings[self.root]
        func = func.replace('nan', 'np.nan').replace('inf', 'np.inf')
            
        return func

    def __Simp(self):
        '''
        Simplify trees by collapsing nodes that have: both children are numbers,
        multiplication / subtraction / addition / power with 0, 1 to a power,
        multiplication by 1, power / division by 1, min / max / subtraction
        with identical children. Must be called, ideally, from Simplify().
        '''
        
        # prepare the iteraion
        levs = list(self.leaves.keys())
        levs.sort
        levs = levs[::-1]
        try:
            levs.remove(0)
        except ValueError:
            # no 0, so just ignore
            pass
        # iterate over each non-root level, and iterate over each leaf to check
        # if it's parent can be simplified; this moves some leaves, and I'm unsure
        # yet of the consequences
        delta = False
        for lev in levs:
            for leaf in self.leaves[lev]:
                # both siblings just numbers - replace the parent, but not if dividing by 0
                if (leaf.parent.left.type == 'const') & (leaf.parent.right.type == 'const') & ~((leaf.parent.value == 'dv') & (leaf.parent.right.value == 0)):
                    # reduce the parent to a new const node
                    newConst = eval(Tree(leaf.parent, 2).GenFunction())
                    leaf.parent.setNode(typ='const', val=newConst)
                    # clear children
                    leaf.parent.setLeft(None)
                    leaf.parent.setRight(None)
                    # remember that we made a change
                    delta = True
                    break
                # mult by 1
                elif (leaf.value == 1) & (leaf.parent.value == 'ml'):
                    # reduce the parent to whichever sibling is not 1, but first check if the parent is a root node:
                    if leaf.parent == self.root:
                        # parent is root, so need to edit the root
                        if leaf.parent.left.value != 1:
                            # point to the left sibling
                            self.root = leaf.parent.left
                            leaf.parent.left.parent = None
                        elif leaf.parent.right.value != 1:
                            # point to the right sibling
                            self.root = leaf.parent.right
                            leaf.parent.right.parent = None
                    else:
                        if leaf.parent.left.value != 1:
                            # figure out if the parent is a left or right
                            if leaf.parent == leaf.parent.parent.left:
                                # point to the left sibling
                                leaf.parent.parent.setLeft(leaf.parent.left)
                            else:
                                # point to the left sibling
                                leaf.parent.parent.setRight(leaf.parent.left)
                            leaf.parent.left.parent = leaf.parent.parent
                        elif leaf.parent.right.value != 1:
                            # figure out of the parent is a left or right
                            if leaf.parent == leaf.parent.parent.left:
                                # point to the right sibling
                                leaf.parent.parent.setLeft(leaf.parent.right)
                            else:
                                # point to the right sibling
                                leaf.parent.parent.setRight(leaf.parent.right)
                            leaf.parent.right.parent = leaf.parent.parent
                    # remember that we made a change
                    delta = True
                    break
                # div / pow by 1 (1 on the right)
                elif (leaf.value == 1) & (leaf.parent.value in ['pw', 'dv'])  & (leaf.parent.right == leaf):
                    # reduce the parent to the left sibling, but first check if the parent is a root node:
                    if leaf.parent == self.root:
                        # parent is root, so need to edit the root
                        self.root = leaf.parent.left
                        leaf.parent.left.parent = None
                    else:
                        # figure out if the parent is a left or right
                        if leaf.parent == leaf.parent.parent.left:
                            # point to the left sibling
                            leaf.parent.parent.setLeft(leaf.parent.left)
                        else:
                            # point to the left sibling
                            leaf.parent.parent.setRight(leaf.parent.left)
                        # set the leaf's parent
                        leaf.parent.left.parent = leaf.parent.parent
                    # remember that we made a change
                    delta = True
                    break
                # pow by 1 (1 on the left)
                elif (leaf.value == 1) & (leaf.parent.value == 'pw')  & (leaf.parent.left == leaf):
                    # just replace with constant 1
                    leaf.parent.setNode(typ='const', val=1)
                    # clear children
                    leaf.parent.setLeft(None)
                    leaf.parent.setRight(None)
                    # remember that we made a change
                    delta = True
                    break
                # mult by 0
                elif (leaf.value == 0) & (leaf.parent.value == 'ml'):
                    # just replace with constant 0
                    leaf.parent.setNode(typ='const', val=0)
                    # remove leaves
                    try:
                        self.leaves[lev].remove(leaf.parent.left)
                        self.leaves[lev].remove(leaf.parent.right)
                    except ValueError:
                        pass
                    # clear children
                    leaf.parent.setLeft(None)
                    leaf.parent.setRight(None)
                    # remember that we made a change
                    delta = True
                    break
                # pow by 0
                elif (leaf.value == 0) & (leaf.parent.value == 'pw'):
                    # just replace with constant 0 or 1 depending on the side
                    if leaf.parent.left == leaf:
                        leaf.parent.setNode(typ='const', val=0)
                    elif leaf.parent.right == leaf:
                        leaf.parent.setNode(typ='const', val=1)
                    # clear children
                    leaf.parent.setLeft(None)
                    leaf.parent.setRight(None)
                    # remember that we made a change
                    delta = True
                    break
                #  add 0
                elif (leaf.value == 0) & (leaf.parent.value == 'ad'):
                    # reduce the parent to whichever sibling is not 0, but first check if the parent is a root node:
                    if leaf.parent == self.root:
                        # parent is root, so need to edit the root
                        if leaf.parent.left.value != 0:
                            # point to the left sibling
                            self.root = leaf.parent.left
                            leaf.parent.left.parent = None
                        elif leaf.parent.right.value != 0:
                            # point to the right sibling
                            self.root = leaf.parent.right
                            leaf.parent.right.parent = None
                    else:
                        if leaf.parent.left.value != 0:
                            # figure out of the parent is a left or right
                            if leaf.parent == leaf.parent.parent.left:
                                # point to the left sibling
                                leaf.parent.parent.setLeft(leaf.parent.left)
                            else:
                                # point to the left sibling
                                leaf.parent.parent.setRight(leaf.parent.left)
                            leaf.parent.left.parent = leaf.parent.parent
                        elif leaf.parent.right.value != 0:
                            # figure out of the parent is a left or right
                            if leaf.parent == leaf.parent.parent.left:
                                # point to the left sibling
                                leaf.parent.parent.setLeft(leaf.parent.right)
                            else:
                                # point to the left sibling
                                leaf.parent.parent.setRight(leaf.parent.right)
                            leaf.parent.right.parent = leaf.parent.parent
                    # remember that we made a change
                    delta = True
                    break
                #  subtract 0 (0 on right)
                elif (leaf.value == 0) & (leaf.parent.value == 'sb') & (leaf.parent.right == leaf):
                    # reduce the parent to the left sibling, but first check if the parent is a root node:
                    if leaf.parent == self.root:
                        # parent is root, so need to edit the root
                        self.root = leaf.parent.left
                        leaf.parent.left.parent = None
                    else:
                        # figure out of the parent is a left or right
                        if leaf.parent == leaf.parent.parent.left:
                            # point to the left sibling
                            leaf.parent.parent.setLeft(leaf.parent.left)
                        else:
                            # point to the left sibling
                            leaf.parent.parent.setRight(leaf.parent.left)
                        # set the leaf's parent
                        leaf.parent.left.parent = leaf.parent.parent
                    # remember that we made a change
                    delta = True
                # min, max, or sub the same inputs
                elif (leaf.parent.left.value == leaf.parent.right.value) & (leaf.type != 'op') & (leaf.parent.value in ['mn', 'mx', 'sb']):
                    if leaf.parent.value == 'sb':
                        # just replace with constant 0
                        leaf.parent.setNode(typ='const', val=0)
                        # remove leaves
                        try:
                            self.leaves[lev].remove(leaf.parent.left)
                            self.leaves[lev].remove(leaf.parent.right)
                        except ValueError:
                            pass
                        # clear children
                        leaf.parent.setLeft(None)
                        leaf.parent.setRight(None)
                    else:
                        # replace parent with whatever the siblings are, but first check if the parent is a root node:
                        if leaf.parent == self.root:
                            self.root = leaf
                            leaf.parent = None
                        else:
                            if leaf.parent == leaf.parent.parent.left:
                                # point to the left sibling
                                leaf.parent.parent.setLeft(leaf)
                            else:
                                # point to the left sibling
                                leaf.parent.parent.setRight(leaf)
                            # set the leaf's parent
                            leaf.parent = leaf.parent.parent
                    # remember that we made a change
                    delta = True
                    break
            
            # see if we made any changes
            if delta:
                break
                                
        return delta
    
    def Simplify(self, maxSimp=10):
        '''
        Simplify trees by collapsing nodes that have: both children are numbers,
        multiplication / subtraction / addition / power with 0, 1 to a power,
        multiplication by 1, power / division by 1, min / max / subtraction
        with identical children. This will make several passes of the tree, with
        each pass resulting in at most 1 simplifying change, followed by a call to
        GenStruct().
        :param maxSimp: optional (default=10) maximum number of simplification attempts
        :return sCount: number of simplification attempts made
        '''
        
        # iterate over attemps allowed
        for sCount in range(maxSimp):
            # simplify
            delta = self.__Simp()
            if delta:
                # rebuild
                self.GenStruct()
            else:
                break
                
        return sCount+1


''' functions for building a new tree '''
def BuildTreeRec(currNode, currDepth, maxDepth, nodeMeta):
    '''
    Recursive tree building function; only to be called by BuildTree
    '''

    # exit if too deep or at a leaf
    if (currDepth == maxDepth) or (currNode.type != 'op'):
        return currNode
    
    # hit one short of max depth, so ensure only consts or feats selected
    if currDepth == (maxDepth-1):
        noOpsK = [k for k in nodeMeta.keys() if k != 'op']
        noOpsW = [nodeMeta[t][2] for t in noOpsK]
        nodeTypeL, _ = RandomWeightedSelect(noOpsK, noOpsW, 0)
        nodeTypeR, _ = RandomWeightedSelect(noOpsK, noOpsW, 0)
    else:
        nodeTypeL, _ = RandomWeightedSelect(nodeMeta.keys(), [v[2] for v in nodeMeta.values()], 0)
        nodeTypeR, _ = RandomWeightedSelect(nodeMeta.keys(), [v[2] for v in nodeMeta.values()], 0)
        
    # randomly generate the left node
    nodeValuL = nodeMeta[nodeTypeL][0][np.random.randint(nodeMeta[nodeTypeL][1])]
    nodeL = BuildTreeRec(Node(nodeTypeL, nodeValuL, currNode), currDepth+1, maxDepth, nodeMeta)
    currNode.setLeft(nodeL)

    # randomly generate the right node
    nodeValuR = nodeMeta[nodeTypeR][0][np.random.randint(nodeMeta[nodeTypeR][1])]
    nodeR = BuildTreeRec(Node(nodeTypeR, nodeValuR, currNode), currDepth+1, maxDepth, nodeMeta)
    currNode.setRight(nodeR)
    
    return currNode


def BuildTree(maxDepth, nodeMeta, verbose=False):
    '''
    Using a set of types of nodes, build a functional tree.
    :param maxDepth: integer maximum depth allowed for the tree (including the root)
    :param nodeMeta: dictionary holding the a tuple of a list of the node values
        allowed, the number of node values allowed, and node weight for random
        selection; keys are node types of 'ops, 'feat', and 'const'
    :param verbose: optional (default = false) flag to print some info
    :return tree: the complete functional tree
    '''
    
    # if max depth is 1, can't have ops nodes
    if maxDepth == 1:
        # randomly generate the root node type
        noOpsK = [k for k in nodeMeta.keys() if k != 'op']
        noOpsW = [nodeMeta[t][2] for t in noOpsK]
        nodeType, _ = RandomWeightedSelect(noOpsK, noOpsW, 0)
    else:
        # randomly generate the root node type
        nodeType, _ = RandomWeightedSelect(nodeMeta.keys(), [v[2] for v in nodeMeta.values()], 0)
    
    # randomly generate the root node value
    nodeValu = nodeMeta[nodeType][0][np.random.randint(nodeMeta[nodeType][1])]    
    # build the tree
    rootNode = BuildTreeRec(Node(nodeType, nodeValu, None), 0, maxDepth-1, nodeMeta)
    
    return Tree(rootNode, maxDepth)