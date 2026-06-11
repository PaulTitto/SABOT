import os
import xml.etree.ElementTree as ET

GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"


def count_graph_stats(working_dir: str) -> dict:
    """
    Membaca jumlah entities (node) dan relations (edge) dari file .graphml
    """
    graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")

    if not os.path.exists(graphml_path):
        print(f"  ⚠️  File tidak ditemukan: {graphml_path}")
        return {"entities": 0, "relations": 0}

    try:
        tree = ET.parse(graphml_path)
        root = tree.getroot()

        entities = len(root.findall(f".//{GRAPHML_NS}node"))
        relations = len(root.findall(f".//{GRAPHML_NS}edge"))

        return {"entities": entities, "relations": relations}
    except ET.ParseError:
        print(f"Gagal memproses file XML. Pastikan file tidak korup.")
        return {"entities": 0, "relations": 0}

if __name__ == "__main__":
    WORKING_DIR = r"C:\Users\PaulTitto\Downloads\SKRIPSI NLP\final_boss_working_dir"

    graph = count_graph_stats(WORKING_DIR)
    print(graph)
