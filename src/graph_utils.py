from bs4 import BeautifulSoup
from enum import Enum
import re

class NodeTagType(Enum):
    TH = 'th'
    TD = 'td'
    TITLE = 'title'

class Node:
    def __init__(self, value, type:NodeTagType, adj=None):
        self.value = value
        self.type = type
        self.adj = [] if not adj else adj

    def add_adj(self, node):
        self.adj.append(node)

    def print_graph(self):
        level = 0
        queue = [self]

        while queue:
            level_nodes = []
            for _ in range(len(queue)):
                u = queue.pop(0)
                level_nodes.append(u.value)
                for v in u.adj:
                    queue.append(v)

            print(f"Level {level} nodes: {level_nodes}")
            level += 1


def clean_text(text):
    return re.sub(r'\(\s*(hide|show)\s*\)', '', text).strip()

def _extract_top_level_elements(soup, tags):
    res = []
    stack = [soup]

    while stack:
        node = stack.pop()
        valid = False
        for tag in tags:
            if node.name == tag and not node.find_parent(tag):
                valid = valid or True
        if valid:
            res.append(node)
        else:
            stack.extend(node.find_all(recursion=False))

    return res

def _extract_relationship_graph(node_soup):
    res = []
    if node_soup.name == "table":
        table_rows = list(reversed(_extract_top_level_elements(node_soup, ["tr"])))

        cur_node = None
        data_between = False
        for row in table_rows:
            results = _extract_relationship_graph(row)
            if len(results) == 1 and results[0].type == NodeTagType.TH:
                if not cur_node or data_between:
                    cur_node = results[0]
                    res.append(cur_node)
                    data_between = False
                else:
                    cur_node.value += results[0].value
            else:
                data_between = True
                for result in results:
                    if result.type == NodeTagType.TH:
                        if cur_node:
                            cur_node.add_adj(result)
                        else:
                            res.append(result)

    elif node_soup.name == "tr":
        elements = _extract_top_level_elements(node_soup, ["th", "th_colspan", "table", "td"])
        for element in elements:
            if element.name == "th" or element.name == "th_colspan":
                res.append(Node(clean_text(element.text), NodeTagType.TH))
            elif element.name == "td":
                res.append(Node(clean_text(element.text), NodeTagType.TD))
            else:
                res.append(_extract_relationship_graph(element))
    return res

def extract_relationship_graphs(simple_text: str):
    soup = BeautifulSoup(simple_text, "lxml")
    title = soup.find("h1")
    if not title:
        raise Exception(f"Could not find title of page with simple text: {simple_text}.")

    root = Node(clean_text(title.text), NodeTagType.TITLE)
    for table in _extract_top_level_elements(soup, ["table"]):
        table_nodes = _extract_relationship_graph(table)
        for node in table_nodes:
            root.add_adj(node)

    return root
