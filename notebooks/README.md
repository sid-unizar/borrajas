# Notebooks

A collection of Jupyter notebooks with experiments and demonstrations of the RDF-aware LLM approaches.

### `borrajas_feedback_agentic.ipynb`

This notebook implements a pipeline with a full agent in action for a question answering task in the climate domain. This is divided into a router, a SPARQL generator, a synthesizer, and a feedback component. The used LLM can be different for those tasks, designed as nodes of the langgraph workflow. The feedback node is set at the end of the pipeline, and approves or rejects the final answer. If approved, the final answer is presented to the user, if rejected, the agentic loop starts again from the beginning. The pipeline has access to two knowledge sources: a knowledge graph for RDF-aware reasoning and a vector database for the RAG component.

### `borrajas_agentic_no_feedback.ipynb`

This notebook implements a pipeline that follows the same schema of `borrajas_feedback_agentic.ipynb`, but the feedback component is taken out. In this way, the final answer is not evaluated by the agent, but it is directly presented to the user.

### `borrajas_feedback_deterministic.ipynb`

This notebook implements a pipeline where the agentic components are reduced with respect to the two above. The KG exploration and the RAG component are not anymore left to the LLM's choice, but they are made deterministic. The agentic component remains present in the SPARQL generation and in the feedback node. 

### `borrajas_deterministic_no_feedback.ipynb`

This notebook is similar to `borrajas_feedback_determinstic.ipynb`, with the only difference of removing the feedback loop.

### `borrajas_agentic_fixed_ordered.ipynb`

This notebook implements a pipeline where the decision of going first to the knowledge graph or to the RAG component is not left to the router agent. A fixed order can be chosen by using two differently shaped graphs, one where the KG node comes first and one where the RAG one does. The feedback node is maintaned unchanged. 
