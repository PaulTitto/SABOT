# import os
# import shutil
# def main():
#     WORKING_DIR = "./gemini_rag_storage"
#     files_to_delete = [
#         "graph_chunk_entity_relation.graphml",
#         "kv_store_doc_status.json",
#         "kv_store_full_docs.json",
#         "kv_store_text_chunks.json",
#         "vdb_chunks.json",
#         "vdb_entities.json",
#         "vdb_relationships.json",
#     ]
#
#     for file in files_to_delete:
#         file_path = os.path.join(WORKING_DIR, file)
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"Deleting old file:: {file_path}")
#
#     if os.path.exists(WORKING_DIR):
#         shutil.rmtree(WORKING_DIR)
#         print("Storage cleared for a fresh start.")
#
# if __name__ == "__main__":
#     main()