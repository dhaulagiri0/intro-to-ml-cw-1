# Follows the buchheim algorithm:
# Source: https://llimllib.github.io/pymag-trees/

class DrawTree(object):
    def __init__(self, tree, parent=None, depth=0, left_brother=None):
        self.x = -1.
        self.y = depth
        self.tree = tree

        self.children = []
        self.left = None
        self.right = None

        if not isinstance(tree, dict):
            self.label = str(tree)
        else:
            self.label = f"[X{tree['feature_index']} < {tree['feature_threshold']:.2f}]"
            if tree_left := tree["left_tree"]:
                self.left = DrawTree(tree_left, self, depth + 1)
                self.children.append(self.left)
            if tree_right := tree["right_tree"]:
                self.right = DrawTree(tree_right, self, depth + 1, self.left)
                self.children.append(self.right)

        self.parent = parent
        self.thread = None
        self.mod = 0
        self.change = self.shift = 0
        self.left_brother = left_brother

    def next_left(self):
        return self.thread or self.left

    def next_right(self):
        return self.thread or self.right


def buchheim(tree):
    dt = firstwalk(DrawTree(tree))
    second_walk(dt)
    return dt

DISTANCE = 1.2  

def firstwalk(cur):
    """
    Does a post-order traversal of the tree to set initial x-coordinates of the children.
    Then place parent at midpoint of its children
    """
    if not cur.children:
        if cur.left_brother:
            cur.x = cur.left_brother.x + DISTANCE
        else:
            cur.x = 0
    else:
        for c in cur.children:
            firstwalk(c)
        if cur.right and cur.left:
            apportion(cur)
        execute_shifts(cur)
        midpoint = (cur.children[0].x + cur.children[-1].x) / 2

        left_brother = cur.left_brother
        if left_brother:
            cur.x = left_brother.x + DISTANCE
            cur.mod = cur.x - midpoint
        else:
            cur.x = midpoint
    return cur

def apportion(cur):
    """
    Iterates through the "contours" of the left and right subtrees to check overlaps and space them accordingly
    Creates thread pointers between levels of the tree where necessary to speed up computations
    """
    if not cur.right or not cur.left:
        return

    # in buchheim notation:
    # i == inner; o == outer; r == right; l == left;
    cir = cor = cur.right
    cil = col = cur.left
    sir = sor = cur.right.mod
    sil = cur.left.mod
    sol = cur.left.mod

    while cil and cir and cil.next_left() and cir.next_right():
        cil = cil.next_right()
        cir = cir.next_left()
        if col:
            col = col.next_left()
        if cor:
            cor = cor.next_right()

        shift = (cil.x + sil) - (cir.x + sir) + DISTANCE
        if shift > 0:
            move_subtree(cur.left, cur.right, shift)
            sir = sir + shift
            sor = sor + shift

        sil += cil.mod
        sir += cir.mod
        if col:
            sol += col.mod
        if cor:
            sor += cor.mod

    if cil and cil.right and cor and not cor.right:
        cor.thread = cil.right
        cor.mod += sil - sor
    elif cir and cir.left and col and not col.left:
        col.thread = cir.left
        col.mod += sir - sol


def move_subtree(wl, wr, shift):
    wr.change -= shift
    wr.shift += shift
    wl.change += shift
    wr.x += shift
    wr.mod += shift


def execute_shifts(cur):
    shift = change = 0
    for c in cur.children[::-1]:
        c.x += shift
        c.mod += shift
        change += c.change
        shift += c.shift + change


def second_walk(cur, m=0):
    """
    Pre-order traversal of the tree to finalize x-coordinates by using the mod values
    """
    cur.x += m

    for c in cur.children:
        second_walk(c, m + cur.mod)
