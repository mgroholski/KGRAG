from bs4 import BeautifulSoup
from enum import Enum
import re

class NodeTagType(Enum):
    TH = 'th'
    TH_COLSPAN = 'th_colspan'
    TD = 'td'
    TITLE = 'title'
    H2 = 'h2'
    H3 = 'h3'

class Node:
    def __init__(self, value, type:NodeTagType, adj=None):
        self.value = value
        self.type = type
        self.adj = [] if not adj else adj
        self.data = []

    def add_adj(self, node):
        self.adj.append(node)

    def add_data(self, data):
        self.data.append(data)

    def print_graph(self):
        level = 0
        queue = [self]

        while queue:
            level_nodes = []
            for _ in range(len(queue)):
                u = queue.pop(0)
                level_nodes.append((u.value, u.data))
                for v in u.adj:
                    queue.append(v)

            print(f"Level {level} nodes: {level_nodes}")
            level += 1

def _clean_text(text):
    return re.sub(r'\(\s*(hide|show|edit)\s*\)', '', text).strip()

def _extract_first_level_tags(soup, tags):
    found_tags = soup.find_all(tags)
    first_level_tags = []
    seen_tags = set()

    for tag in found_tags:
        if tag.find_parents("nav"):  # Exclude if inside a <nav>
            continue
        if not any(parent in seen_tags for parent in tag.find_parents()):
            first_level_tags.append(tag)
            seen_tags.add(tag)

    return first_level_tags

def _extract_table_graph(soup, root=None):
    cur_parent = root

    table_rows = _extract_first_level_tags(soup, ["tr"])
    col_headings = []

    data_between = True
    for row in table_rows:
        row_table = row.find("table")
        if row_table:
            heading = row.find("h3")
            head_node = cur_parent
            if heading:
                head_node = Node(_clean_text(heading.text), NodeTagType.H3)
                if cur_parent:
                    cur_parent.add_adj(head_node)
                else:
                    root = head_node
                    cur_parent = head_node
            _extract_table_graph(row_table, root=head_node)
            continue

        row_elements = row.find_all(["th", "td", "th_colspan"])
        if len(row_elements) == 1:
            element = row_elements[0]
            if element.name == "th_colspan":
                # Cleans out column headings if new table.
                if len(col_headings):
                    if cur_parent:
                        for col_heading in col_headings:
                            cur_parent.add_adj(col_heading)
                    else:
                        print(f"Found col_headings before parent: {soup}")
                        exit()

                if data_between:
                    head_node = Node(_clean_text(element.text), NodeTagType.TH_COLSPAN)
                    if not root:
                        root = head_node
                    else:
                        root.add_adj(head_node)
                    cur_parent = head_node
                    data_between = False
                else:
                    cur_parent.value += " " + _clean_text(element.text)
        else:
            data_between = True
            row_heading = None
            data_idx = 0
            for element in row_elements:
                if element.name == "th" or element.name == "th_colspan":
                    if not row_heading:
                        row_heading = Node(_clean_text(element.text), NodeTagType.TH if element.name=="th" else NodeTagType.TH_COLSPAN)
                    else:
                        if not len(col_headings):
                            col_headings.append(row_heading)
                        col_headings.append(Node(_clean_text(element.text), NodeTagType.TH if element.name=="th" else NodeTagType.TH_COLSPAN))
                elif element.name == "td":
                    if len(col_headings):
                        col_headings[data_idx].data.append(element.text)
                        data_idx = ((data_idx + 1) % len(col_headings))
                    else:
                        if row_heading:
                            row_heading.data.append(element.text)
                        elif cur_parent:
                            cur_parent.data.append(element.text)

            if not len(col_headings) and row_heading:
                if cur_parent:
                    cur_parent.add_adj(row_heading)
                else:
                    print(f"Adding row heading without parent: {soup}.")
                    exit()

    # Cleans out column headings if no new heading.
    if len(col_headings):
        if cur_parent:
            for col_heading in col_headings:
                cur_parent.add_adj(col_heading)
        else:
            print(f"End: Found col_headings before parent: {soup}")
            exit()

    return root

def extract_relationship_graphs(simple_text: str):
    # Extracts title from page
    soup = BeautifulSoup(simple_text, "lxml").html
    title = soup.find("h1")
    if not title:
        raise Exception(f"Could not find title of page with simple text: {simple_text}.")

    root = Node(_clean_text(title.text), NodeTagType.TITLE)

    first_level_tags = _extract_first_level_tags(soup, ["h2", "h3", "table"])

    cur_table_section = root
    cur_section = root
    for tag in first_level_tags:
        if tag.name == "h2":
            node_text = _clean_text(tag.text)
            if node_text:
                n = Node(node_text, NodeTagType.H2)
                root.adj.append(n)
                cur_section = n
                cur_table_section = n
        elif tag.name == "h3":
            node_text = _clean_text(tag.text)
            if node_text:
                n = Node(node_text, NodeTagType.H3)
                cur_section.add_adj(n)
                cur_table_section = n
        else:
            _extract_table_graph(tag, cur_table_section)

    return root
