from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex


InfoDocs = SimpleDirectoryReader("./InfoDocs").load_data()


Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
)
infoNodes = splitter.get_nodes_from_documents(InfoDocs)


storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore(),
    vector_store=SimpleVectorStore(),
    index_store=SimpleIndexStore(),
)

storage_context.docstore.add_documents(infoNodes)
storage_context.persist(persist_dir="./infoStorage")

vector_index = VectorStoreIndex(infoNodes, storage_context=storage_context)
vector_index.set_index_id("1")
vector_index.storage_context.persist(persist_dir="./infoStorage")

keyword_index = SimpleKeywordTableIndex(infoNodes, storage_context=storage_context)
keyword_index.set_index_id("2")
keyword_index.storage_context.persist(persist_dir="./infoStorage")
