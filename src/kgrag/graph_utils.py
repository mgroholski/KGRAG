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
    ARRAY_TABLE = 'arr_table'

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
    text = re.sub(r'\[\s*(hide|show|edit)\s*\]', '', text).strip()
    text = re.sub(r'\n', ', ', text)
    return text

def _clean_int(text)->int:
    return int(re.sub(r'[^0-9]', '', text))

def extract_first_level_tags(soup, tags):
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

    table_rows = extract_first_level_tags(soup, ["tr"])
    col_headings = []

    data_between = True
    found_head = False


    def flush_table(cur_parent):
        # Cleans out column headings if no new heading.
        if len(col_headings):
            if cur_parent:
                node_headings = []
                node_rows = []
                for col_heading in col_headings:
                    node_headings.append(col_heading.value)
                    if not len(node_rows):
                        for data in col_heading.data:
                            node_rows.append([])
                            for d in data:
                                node_rows[-1].append(d)
                    else:
                        for idx, data in enumerate(col_heading.data):
                            for d in data:
                                node_rows[idx].append(d)
                cur_parent.data.append({"column headings": node_headings, "rows": node_rows})
                cur_parent.type = NodeTagType.ARRAY_TABLE
            else:
                print(f"End: Found col_headings before parent: {soup}")
                exit()

    row_store = []
    head_span = []
    max_header_row_span = 1
    data_seen = False
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

        # Assign elements with th and colspan > 1 to th_colspan name
        row_elements = row.find_all(["th", "td"])
        for element in row_elements:
            if element.name == "th" and element.get("colspan") and _clean_int(element.get("colspan")) > 1:
                element.name = "th_colspan"

        if not len(col_headings) and len(row_elements) == 1:
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
                    # Use root from this function call.
                    if not found_head:
                        root = head_node
                        found_head = True
                    else:
                        root.add_adj(head_node)
                    cur_parent = head_node
                    data_between = False
                else:
                    cur_parent.value += " " + _clean_text(element.text)
        else:
            # Use root from last function call.
            found_head = True
            data_between = True

            row_heading = None
            row_tags = set([(element.name if element.name != "th_colspan" and _clean_text(element.text).lower() != "n/a" else "th") for element in row_elements])
            if len(row_elements):
                if row_elements[0].name == "th" and len(row_tags) > 1:
                    element = row_elements[0]
                    row_heading = Node(_clean_text(element.text), NodeTagType.TH if element.name=="th" else NodeTagType.TH_COLSPAN)
                    for element in row_elements[1:]:
                        row_heading.data.append(_clean_text(element.text))
                else:
                    if len(row_tags) == 1 and "th" in row_tags:
                        # if data in between flush it
                        if data_seen:
                            flush_table(cur_parent)
                            row_store = []
                            data_seen = False
                            col_headings = []

                        for element in row_elements:
                            if element.get("rowspan"):
                                max_header_row_span = max(max_header_row_span, _clean_int(element.get("rowspan")))

                        head_idx = 0
                        for element in row_elements:
                            row_span = 1
                            if element.get("rowspan"):
                                row_span = _clean_int(element.get("rowspan"))
                            col_span = 1
                            if element.get("colspan"):
                                col_span = _clean_int(element.get("colspan"))
                            for _ in range(col_span):
                                if head_idx < len(head_span):
                                    # Skips if current head column has max row span
                                    while head_idx < len(head_span) and head_span[head_idx][0] == max_header_row_span:
                                        head_idx += 1

                                    if head_idx == len(head_span):
                                        head_span.append((row_span, _clean_text(element.text)))
                                    else:
                                        head_span[head_idx] = (head_span[head_idx][0] + row_span, head_span[head_idx][1] + f" {_clean_text(element.text)}")
                                else:
                                    head_span.append((row_span, _clean_text(element.text)))
                                head_idx += 1
                    else:
                        for _, head in head_span:
                            col_headings.append(Node(head, NodeTagType.TH if element.name=="th" else NodeTagType.TH_COLSPAN))
                        max_header_row_span = 1
                        head_span = []
                        data_seen = True

                        if len(col_headings):
                            # Creates row_store if it doesn't exist yet.
                            if not len(row_store):
                                row_store = [[] for _ in range(len(col_headings))]

                            for i in range(len(row_store)):
                                col_headings[i].data.append([])
                                for _ in range(len(row_store[i])):
                                    cnt, txt = row_store[i].pop(0)
                                    col_headings[i].data[-1].append(txt)
                                    if cnt > 1:
                                        row_store[i].append((cnt-1, txt))

                        idx = 0
                        last_idx = False
                        for element in row_elements:
                            if len(col_headings):
                                n_headings = len(col_headings)

                                row_span = 1
                                if element.get("rowspan"):
                                    row_span = _clean_int(element.get("rowspan"))

                                col_span = 1
                                if element.get("colspan"):
                                    col_span = _clean_int(element.get("colspan"))

                                element_text = _clean_text(element.text)
                                # If data spans multiple columns, add that data to the columns.
                                for _ in range(col_span):
                                    while len(col_headings[idx].data[-1]) and not last_idx:
                                        idx = min(idx + 1, n_headings - 1)
                                        last_idx = (idx == (n_headings - 1))

                                    col_headings[idx].data[-1].append(element_text)

                                    # Put multi-row data back into row_store if it still has rows.
                                    if row_span > 1:
                                        row_store[idx].append((row_span-1, element_text))
                                    idx = min(idx + 1, n_headings - 1)
                            else:
                                if row_heading:
                                    row_heading.data.append(_clean_text(element.text))
                                elif cur_parent:
                                    cur_parent.data.append(_clean_text(element.text))
            if not len(col_headings) and row_heading:
                if cur_parent:
                    cur_parent.add_adj(row_heading)
                else:
                    print(f"Adding row heading without parent: {soup}.")
                    exit()

    # Cleans out column headings if no new heading.
    flush_table(cur_parent)
    return root

def extract_relationship_graphs(simple_text: str):
    # Extracts title from page
    soup = BeautifulSoup(simple_text, "lxml").html
    title = soup.find("h1")
    if not title:
        raise Exception(f"Could not find title of page with simple text: {simple_text}.")

    root = Node(_clean_text(title.text), NodeTagType.TITLE)

    first_level_tags = extract_first_level_tags(soup, ["h2", "h3", "table"])

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
            subroot = _extract_table_graph(tag, cur_table_section)
            if subroot != cur_table_section:
                cur_table_section.add_adj(subroot)

    return root
